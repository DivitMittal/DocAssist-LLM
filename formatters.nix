{inputs, ...}: {
  imports = [
    inputs.treefmt-nix.flakeModule
  ];

  perSystem.treefmt = {
    flakeCheck = false;

    programs = {
      #typos.enable = true;
      ## Nix
      alejandra.enable = true;
      deadnix.enable = true;
      statix.enable = true;
      ## Python
      black.enable = true;
    };

    projectRootFile = "flake.nix";
  };
}