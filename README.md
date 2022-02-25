# VampireSurvivors A.I.
<img src="./main.png">

##### VampireSurvivors A.I.는 최근 유행하는 로그라이크 장르게임 중 하나인 Vampire Survivors에 실전성 있는 Deep-RL을 접목하여 설계한 강화학습 프로젝트이다.
##### 프로젝트에 강화학습을 실제로 적용하는 것은 생각보다 까다로운데, 해당 프로젝트의 setting을 어떻게 구성할지, 또 reward는 어떻게 정할지에 따라 해당 강화학습 프로젝트의 학습 성능이 크게 갈리는 경향이 있기 때문이다.
##### 그래서 위 게임을 클리어하기 위해선 두가지 과제를 해결해야 했는데, 게임 내 화면에 나오는 몹들에 대한 agent의 방향키 조절,그리고 레벨업 상황에서 아이템 선택을 정하는 것이다. 

#### *Read this in other languages: [to English](README.eng.md)

## PPO
#### ● 방향키 조절에 대한 알고리즘은 해당 게임이 Grid World 라는 것을 감안하여 레벨업, 상자 획득과 기본 시간 생존을 양수 값으로 설정하였고, 죽는 화면 근처 행동값에 대한 보상에 음수 값을 추가 하였다.
#### ● Grid World를 탐색하는 알고리즘 중에 2017년도에 공개된  PPO(Proximal Policy Optimization)는 이전 행동과 현재 행동을 비교하여 loss값을 구성하는 알고리즘으로 Grid World 학습에 낮은 Episode를 적용해도 뛰어난 학습 성능을 보여줬다.
#### ● 위 게임의 장르 특성상 장시간 생존으로 넘어가게 되면 
