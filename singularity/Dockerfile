Bootstrap: docker
From: condaforge/miniforge3

%post

    # Set proxy for the environment
    export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
    export HTTP_PROXY="http://www-cache.gwdg.de:3128"

    # Ensure wget uses the proxy by setting it explicitly in the environment
    export http_proxy="http://www-cache.gwdg.de:3128"
    export https_proxy="http://www-cache.gwdg.de:3128"

    # Set proxy for apt directly in its config file
    echo 'Acquire::http::Proxy "http://www-cache.gwdg.de:3128";' > /etc/apt/apt.conf
    echo 'Acquire::https::Proxy "http://www-cache.gwdg.de:3128";' >> /etc/apt/apt.conf

    export DEBIAN_FRONTEND=noninteractive 
    apt update 
    apt upgrade -y

    conda install --quiet --yes \
        'notebook=7.2.1' \
        'jupyterhub=4.1.5' \
        'jupyterlab=4.2.3'
   
    conda install --quiet --yes --channel pytorch --channel nvidia \
        pytorch \
        pytorch-cuda=12.1 \
        torchaudio \
        torchvision 