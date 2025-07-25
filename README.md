# AmericaStockHunters-BE
>프로메테우스 7기 - 프메개미: 미국주식 자동매매 프로젝트 웹 백엔드 입니다.
---
## Quick Start
``` bash
conda create {your-env} python=3.11 -y
conda activate {your-env}

pip install fastapi
pip install "uvicorn[standard]"

git clone git@github.com:prometheus-11team/AmericaStockHunters-BE.git
cd AmericaStockHunters-BE

uvicorn main:app --reload
```
---
### Check with UI
``` bash
cd ..
git clone git@github.com:prometheus-11team/AmericaStockHunters-FE.git
cd AmericaStockHunters-FE

# make sure the installation of Node.js (including npm)
npm install
npm start
```