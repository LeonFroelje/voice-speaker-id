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
          cfg = config.services.speaker-api;
          defaultPkg = self.packages.${pkgs.system}.default;
        in
        {
          options.services.speaker-api = with lib; {
            enable = lib.mkEnableOption "Speaker Identification API Server";

            package = lib.mkOption {
              type = lib.types.package;
              default = defaultPkg;
              description = "The Speaker API package to use.";
            };

            host = lib.mkOption {
              type = lib.types.str;
              default = "127.0.0.1";
              description = "Hostname or IP to bind the server to.";
            };

            port = lib.mkOption {
              type = lib.types.int;
              default = 8001;
              description = "Port for the FastAPI server.";
            };

            modelPath = lib.mkOption {
              type = lib.types.str;
              default = "/var/lib/speaker-api/cam++.onnx";
              description = "Path to the CAM++ ONNX model file.";
            };
          };

          config = lib.mkIf cfg.enable {
            systemd.services.speaker-api = {
              description = "Speaker Identification FastAPI Service";
              wantedBy = [ "multi-user.target" ];
              after = [ "network.target" ];

              environment = {
                SPEAKER_HOST = cfg.host;
                SPEAKER_PORT = toString cfg.port;
                SPEAKER_MODEL_PATH = cfg.modelPath;
                SPEAKER_DB_PATH = "/var/lib/speaker-api/embeddings.json";

                # CRITICAL FOR CUDA
                LD_LIBRARY_PATH = "/run/opengl-driver/lib";
                PYTHONUNBUFFERED = "1";
              };

              serviceConfig = {
                ExecStart = "${pkgs.writeShellScript "start-speaker-api" ''
                  exec ${cfg.package}/bin/uvicorn speaker_api.main:app --host ''${SPEAKER_HOST} --port ''${SPEAKER_PORT}
                ''}";

                # State Management für embeddings.json und Modelle
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
                  "/dev/nvidia-uvm-tools rwm"
                  "/dev/nvidia-modeset rwm"
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
