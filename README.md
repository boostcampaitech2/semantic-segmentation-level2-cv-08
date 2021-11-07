# pstage_02_Object_Segmentation
---
## Getting Started
---
# MMSegmentation setting
```
git clone https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-08.git .

conda create -n segmentation python=3.8 -y

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y

pip install -r requirements.txt # smp 0.1.3

pip install pytorch-segmentation-model # smp 0.2.0
```


# 🌏재활용 품목 분류를 위한 Semantic segmentation Report🌏

# 목차

- [팀소개](#팀소개)
- [대회 개요](#대회-개요)
- [문제 정의](#문제-정의)
- [실험 내용](#문제에-대한-실험)
- [Modeling 및 Ensemble](#Modeling-및-Ensemble)
- [회고](#회고)

# 팀소개


<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Seoheesu1">
        <img src="https://avatars.githubusercontent.com/u/63832160?v=4" width="100px;" alt=""/>
        <br />
        <sub>서희수</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/WonsangHwang">
        <img src="https://avatars.githubusercontent.com/u/49892621?v=4" width="100px;" alt=""/>
        <br />
        <sub>황원상</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/sala0320">
        <img src="https://avatars.githubusercontent.com/u/49435163?v=4" width="100px;" alt=""/>
        <br />
        <sub>조혜원</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hongsusoo">
        <img src="https://avatars.githubusercontent.com/u/77658029?v=4" width="100px;" alt=""/>
        <br />
        <sub>홍요한</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Junhyuk93">
        <img src="https://avatars.githubusercontent.com/u/61610411?v=4" width="100px;" alt=""/>
        <br />
        <sub>박준혁</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hanlyang0522">
        <img src="https://avatars.githubusercontent.com/u/67934041?v=4" width="100px;" alt=""/>
        <br />
        <sub>박범수</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/GunwooHan">
        <img src="https://avatars.githubusercontent.com/u/76226252?v=4" width="100px;" alt=""/>
        <br />
        <sub>한건우</sub>
      </a>
    </td>
  </tr>
  <tr>
    </td>
  </tr>
</table>
<br>  

# 대회 개요

![](https://i.imgur.com/PnOdQ0L.png)

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

- **Input :** 쓰레기 객체가 담긴 이미지와 bbox 정보(좌표, 카테고리)가 모델의 인풋으로 사용됩니다. bbox annotation은 COCO format으로 제공됩니다. (COCO format에 대한 설명은 학습 데이터 개요를 참고해주세요.)
- **Output** : 모델은 bbox 좌표, 카테고리, score 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다. (submission format에 대한 설명은 평가방법을 참고해주세요

# 문제 정의


# 문제에 대한 실험


# Modeling 및 Ensemble

# 회고
