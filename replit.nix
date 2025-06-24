{ pkgs }: {
    deps = [
        pkgs.python310Full
        pkgs.python310Packages.pip
        pkgs.python310Packages.flask
        pkgs.python310Packages.gunicorn
    ];
} 