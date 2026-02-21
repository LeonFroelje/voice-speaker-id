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
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
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
        fastapi
        uvicorn
        python-multipart
        pydantic
        setuptools
        pydantic-settings
        onnxruntime
        torch
        torchcodec
        torchaudio
        numpy
        soundfile
      ];
      wespeaker-model = pkgs.fetchurl {
        url = "https://wenet.org.cn/downloads?models=wespeaker&version=voxceleb_resnet293_LM.onnx";
        # You may need to update this hash if the download fails
        sha256 = "sha256-5m+R7z6K0YFmXvXqI5m6UuY0K6LpI1vXpI1vXpI1vXp=";
      };
    in
    {
      packages.${system} = {
        default = python.pkgs.buildPythonApplication {
          pname = "speaker-api";
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
            enable = mkEnableOption "Speaker Identification API Server";
            package = mkOption {
              type = types.package;
              default = self.package.default;
            };
            host = mkOption {
              type = types.str;
              default = "127.0.0.1";
            };
            port = mkOption {
              type = types.int;
              default = 8001;
            };
            # Now defaults to the fetched nix store path
            modelPath = mkOption {
              type = types.path;
              default = wespeaker-model;
              description = "Path to the ONNX model file.";
            };
          };

          config = lib.mkIf cfg.enable {
            systemd.services.voice-speaker-id = {
              description = "Speaker Identification FastAPI Service";
              wantedBy = [ "multi-user.target" ];
              after = [ "network.target" ];

              environment = {
                SPEAKER_HOST = cfg.host;
                SPEAKER_PORT = toString cfg.port;
                SPEAKER_MODEL_PATH = toString cfg.modelPath;
                SPEAKER_DB_PATH = "/var/lib/speaker-api/embeddings.json";
                LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.linuxPackages.nvidia_x11}/lib";
                PYTHONUNBUFFERED = "1";
              };

              serviceConfig = {
                ExecStart = "${cfg.package}/bin/uvicorn speaker_api.main:app --host \${SPEAKER_HOST} --port \${SPEAKER_PORT}";
                StateDirectory = "speaker-api";
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
