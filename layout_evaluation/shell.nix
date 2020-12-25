{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    nativeBuildInputs = with pkgs.python38Packages; [
      pandas
      cycler
      greenlet
      matplotlib
      kiwisolver
      msgpack
      numpy
      pillow
      pyparsing
      python-dateutil
      pytz
      six
      tabulate
    ];
  }
