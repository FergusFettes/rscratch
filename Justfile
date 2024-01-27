install_neel_plotly:
  poetry shell
  pip install git+https://github.com/neelnanda-io/neel-plotly.git

install_poetry:
  apt update && apt install -y python3-pip
  pip3 install poetry
