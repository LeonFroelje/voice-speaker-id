{
  description = "Speaker Identification Microservice";

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos-cuda.org"
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.11";
    nixvim = {
      url = "github:nix-community/nixvim";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixvimModules = {
      url = "github:LeonFroelje/nixvim-modules";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      nixvim,
      nixvimModules,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      python = pkgs.python313; # 3.11 ist oft sicherer für Torch/ONNX in Nix
      apiDependencies = with python.pkgs; [
        pydantic
        setuptools
        pydantic-settings
        onnxruntime
        torch
        torchcodec
        torchaudio
        numpy
        soundfile
        aiomqtt
        boto3
      ];
      wespeaker-model = ./voxblink2_samresnet100_ft.onnx;
    in
    {
      packages.${system} = {
        default = python.pkgs.buildPythonApplication {
          pname = "voice-speaker-id";
          version = "0.1.0";
          pyproject = true;
          src = ./.;

          propagatedBuildInputs = apiDependencies;
        };
      };

      nixosModules.default =
        {
          config,
          lib,
          pkgs,
          ...
        }:
        let
          cfg = config.services.voiceSpeakerId;
        in
        {
          options.services.voiceSpeakerId = with lib; {
            enable = mkEnableOption "Speaker Identification Worker";

            package = mkOption {
              type = types.package;
              default = self.packages.${pkgs.system}.default;
              description = "The Speaker ID worker package to use.";
            };

            environmentFile = mkOption {
              type = types.nullOr types.path;
              default = null;
              description = ''
                Path to an environment file for secrets/overrides.
                To prevent leaks, this file should contain:
                - SPEAKER_S3_SECRET_KEY
                - SPEAKER_MQTT_PASSWORD (if your broker requires auth)
              '';
            };

            # --- MQTT Connection ---
            mqttHost = mkOption {
              type = types.str;
              default = "localhost";
              description = "Mosquitto broker IP/Hostname";
            };

            mqttPort = mkOption {
              type = types.int;
              default = 1883;
              description = "Mosquitto broker port";
            };

            mqttUser = mkOption {
              type = types.nullOr types.str;
              default = null;
              description = "Username used to authenticate with MQTT broker";
            };

            # --- Object Storage (S3 Compatible) ---
            s3Endpoint = mkOption {
              type = types.str;
              default = "http://localhost:3900";
              description = "URL to S3 storage";
            };

            s3AccessKey = mkOption {
              type = types.str;
              default = "your-access-key";
              description = "S3 Access Key";
            };

            s3Bucket = mkOption {
              type = types.str;
              default = "voice-commands";
              description = "S3 Bucket Name";
            };

            # --- Model & Data Settings ---
            modelPath = mkOption {
              type = types.path; # or types.str if you prefer
              default = wespeaker-model; # Keeping your original default
              description = "Path to the ONNX model file.";
            };

            dataDir = mkOption {
              type = types.path;
              default = "/var/lib/${self.packages.${system}.default.pname}";
            };

            dbPath = mkOption {
              type = types.str;
              default = "${cfg.dataDir}/embeddings.json";
              description = "Path to the JSON database storing speaker embeddings.";
            };

            # --- Algorithm Settings ---
            threshold = mkOption {
              type = types.float;
              default = 0.5;
              description = "Cosine similarity threshold (0.0 to 1.0)";
            };

            emaAlpha = mkOption {
              type = types.float;
              default = 0.05;
              description = "Exponential Moving Average alpha for profile updates";
            };

            # --- System ---
            logLevel = mkOption {
              type = types.str;
              default = "INFO";
              description = "Logging Level (DEBUG, INFO, ERROR)";
            };
          };

          config = lib.mkIf cfg.enable {
            systemd.services.voice-speaker-id = {
              description = "Speaker Identification Worker Service";
              wantedBy = [ "multi-user.target" ];
              after = [ "network.target" ];

              environment =
                let
                  env = {
                    SPEAKER_MQTT_HOST = cfg.mqttHost;
                    SPEAKER_MQTT_PORT = toString cfg.mqttPort;
                    SPEAKER_MQTT_USER = cfg.mqttUser;

                    SPEAKER_S3_ENDPOINT = cfg.s3Endpoint;
                    SPEAKER_S3_ACCESS_KEY = cfg.s3AccessKey;
                    SPEAKER_S3_BUCKET = cfg.s3Bucket;

                    SPEAKER_MODEL_PATH = toString cfg.modelPath;
                    SPEAKER_DB_PATH = cfg.dbPath;

                    SPEAKER_THRESHOLD = toString cfg.threshold;
                    SPEAKER_EMA_ALPHA = toString cfg.emaAlpha;

                    SPEAKER_LOG_LEVEL = cfg.logLevel;

                    LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.linuxPackages.nvidia_x11}/lib";
                    PYTHONUNBUFFERED = "1";
                  };
                in
                lib.filterAttrs (n: v: v != null) env;

              serviceConfig = {
                # Removed the old --host and --port CLI flags, relying purely on Env vars now
                ExecStart = "${cfg.package}/bin/voice-speaker-id";
                EnvironmentFile = lib.mkIf (cfg.environmentFile != null) cfg.environmentFile;

                # Kept as "speaker-api" to prevent losing your existing embeddings.json!
                StateDirectory = "voice-speaker-id";

                DynamicUser = true;
                SupplementaryGroups = [
                  "video"
                  "render"
                ];

                PrivateDevices = false;
                DeviceAllow = [
                  "/dev/nvidia0 rwm"
                  "/dev/nvidiactl rwm"
                  "/dev/nvidia-uvm rwm"
                ];
              };
            };
          };
        };
      devShells.${system} = {
        default =
          (pkgs.buildFHSEnv {
            name = "Python dev shell";
            targetPkgs =
              p: with p; [
                fd
                ripgrep
                ffmpeg
                (python.withPackages (
                  pypkgs: with pypkgs; [
                    fastapi
                    uvicorn
                    python-multipart
                    pydantic
                    pydantic-settings
                    onnxruntime
                    torch
                    torchaudio
                    numpy
                    soundfile
                  ]
                ))
              ];
            runScript = "zsh";
          }).env;
      };
    };
}
