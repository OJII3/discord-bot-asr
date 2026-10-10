{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem = { pkgs, ... }: {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.python312
            pkgs.uv
          ];

          shellHook = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            if [ "$(uname -m)" = "x86_64" ]; then
              export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
          '';
        };
      };
    };
}
