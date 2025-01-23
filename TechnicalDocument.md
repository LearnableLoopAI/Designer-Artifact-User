# **Designer-Artifact-User Multi-agent Active Inference Simulation**

# 1 Summary of entity interactions
The following diagram is a representation of the interactions between the entities in this project. 

![](./designer-artifact-user3.png)

The Designer entity's agent has symbols:

- $a^{\mathrm{DsgArt}}$ is the action from the Designer entity's agent to the Artifact entity's environment segment
- $u^{\mathrm{ArtDsg}}$ is the inferred control states of the Artifact entity's environment segment
- $\theta^{\mathrm{ArtDsg}}$ is the inferred parameters of the Artifact entity's environment segment
- $s^{\mathrm{ArtDsg}}$ is the inferred state of the Artifact entity's environment segment
- $\theta^{\mathrm{UsrDsg}}$ is the inferred parameters of the User entity's environment segment
- $s^{\mathrm{UsrDsg}}$ is the inferred state of the User entity's environment segment

The Designer entity's environment has symbols:

- from Artifact's environment segment:
    - $y^{\mathrm{ArtDsg}}$ is the observation from the Artifact entity's environment segment
    - $\breve\theta^{\mathrm{ArtDsg}}$ is the true parameters of the Artifact entity
    - $\breve s^{\mathrm{ArtDsg}}$ is the true state of the Artifact entity
    - $W^{\mathrm{Art}}$ is the exogenous information impacting the Artifact entity
- from User's environment segment:
    - $y^{\mathrm{UsrDsg}}$ is the observation from the User entity's environment segment
    - $\breve\theta^{\mathrm{UsrDsg}}$ is the true parameters of the User entity
    - $\breve s^{\mathrm{UsrDsg}}$ is the true state of the User entity

The Artifact entity's agent has symbols:

- $a^{\mathrm{ArtUsr}}$ is the action from the Artifact entity's agent to the User entity's environment segment
- $u^{\mathrm{UsrArt}}$ is the inferred control states of the User entity's environment segment
- $\theta^{\mathrm{UsrArt}}$ is the inferred parameters of the User entity's environment segment
- $s^{\mathrm{UsrArt}}$ is the inferred state of the User entity's environment segment

The Artifact entity's environment has symbols:

- from User's environment segment:
    - $y^{\mathrm{UsrArt}}$ is the observation from the User entity's environment segment
    - $\breve\theta^{\mathrm{UsrArt}}$ is the true parameters of the User entity
    - $\breve s^{\mathrm{UsrArt}}$ is the true state of the User entity
    - $W^{\mathrm{Usr}}$ is the exogenous information impacting the User entity

The User entity's agent has symbols:

- $a^{\mathrm{UsrArt}}$ is the action from the User entity's agent to the Artifact entity's environment segment
- $u^{\mathrm{ArtUsr}}$ is the inferred control states of the Artifact entity's environment segment
- $\theta^{\mathrm{ArtUsr}}$ is the inferred parameters of the Artifact entity's environment segment
- $s^{\mathrm{ArtUsr}}$ is the inferred state of the User entity's environment segment

The User entity's environment has symbols:

- from Artifact's environment segment:
    - $y^{\mathrm{ArtUsr}}$ is the observation from the Artifact entity's environment segment
    - $\breve\theta^{\mathrm{ArtUsr}}$ is the true parameters of the Artifact entity
    - $\breve s^{\mathrm{ArtUsr}}$ is the true state of the Artifact entity
    - $W^{\mathrm{Art}}$ is the exogenous information impacting the Artifact entity
  
# 2 Entity interactions

## 2.1 Designer agent
This is the Designer agent's *generative model* for its environment.

### 2.1.1 State factors
The inferred state factors from the Designer's environment are:

- $s^{\mathrm{ArtDsg}}_1$ (EYE_TRACKING)
- $s^{\mathrm{ArtDsg}}_2$ (TIME_ON_TASK)
- $s^{\mathrm{ArtDsg}}_3$ (UI_FEATURES) [LATER]
- $s^{\mathrm{ArtDsg}}_4$ (PERFORMANCE) [LATER]
- $s^{\mathrm{UsrDsg}}_1$ (FEEDBACK) [LATER]

#### 2.1.1.1 $s^{\mathrm{ArtDsg}}_1$ (EYE_TRACKING)
It has two possible values:

- $s^{\mathrm{ArtDsg}}_1 = 0$, `FOCUSED`
- $s^{\mathrm{ArtDsg}}_1 = 1$, `SCATTERED`

#### 2.1.1.2 $s^{\mathrm{Art}}_2$ (TIME_ON_TASK)
It has three possible values:

- $s^{\mathrm{ArtDsg}}_2 = 0$, `SHORT`
- $s^{\mathrm{ArtDsg}}_2 = 1$, `MEDIUM`
- $s^{\mathrm{ArtDsg}}_2 = 2$, `LONG`

#### 2.1.1.3 $s^{\mathrm{ArtDsg}}_3$ (UI_FEATURES) [LATER]
It has three possible values: 

- $s^{\mathrm{ArtDsg}}_3 = 0$, `MINIMAL`
- $s^{\mathrm{ArtDsg}}_3 = 1$, `STANDARD`
- $s^{\mathrm{ArtDsg}}_3 = 2$, `ADVANCED`

#### 2.1.1.4 $s^{\mathrm{ArtDsg}}_4$ (PERFORMANCE) [LATER]
It has three possible values:

- $s^{\mathrm{ArtDsg}}_4 = 0$, `LOW`
- $s^{\mathrm{ArtDsg}}_4 = 1$, `MEDIUM`
- $s^{\mathrm{ArtDsg}}_4 = 2$, `HIGH`

#### 2.1.1.5 $s^{\mathrm{UsrDsg}}_1$ (FEEDBACK) [LATER]
It has two possible values:

- $s^{\mathrm{UsrDsg}}_1 = 0$, `POSITIVE`
- $s^{\mathrm{UsrDsg}}_1 = 1$, `NEGATIVE`

### 2.1.2 Observation modalities 
The observed observation modalities from the Designer's environment (inputs to the Designer agent) are:

- $y^{\mathrm{ArtDsg}}_1$ (EYE_TRACKING_OBS)
- $y^{\mathrm{ArtDsg}}_2$ (TIME_ON_TASK_OBS)
- $y^{\mathrm{ArtDsg}}_3$ (UI_FEATURES_OBS)
- $y^{\mathrm{ArtDsg}}_4$ (PERFORMANCE_OBS) [LATER]
- $y^{\mathrm{UsrDsg}}_1$ (FEEDBACK_OBS) [LATER]

#### 2.1.2.1 $y^{\mathrm{ArtDsg}}_1$ (EYE_TRACKING_OBS)
It has three possible values:

- $y^{\mathrm{ArtDsg}}_1 = 0$, `FOCUSED_OBS`
- $y^{\mathrm{ArtDsg}}_1 = 1$, `SCATTERED_OBS`
- $y^{\mathrm{ArtDsg}}_1 = 2$, `NEUTRAL_OBS`

#### 2.1.2.2 $y^{\mathrm{ArtDsg}}_2$ (TIME_ON_TASK_OBS)
It has three possible values:

- $y^{\mathrm{ArtDsg}}_2 = 0$, `SHORT_OBS`
- $y^{\mathrm{ArtDsg}}_2 = 1$, `MEDIUM_OBS`
- $y^{\mathrm{ArtDsg}}_2 = 2$, `LONG_OBS`

#### 2.1.2.3 $y^{\mathrm{ArtDsg}}_3$ (UI_FEATURES_OBS)
It has three possible values: 

- $y^{\mathrm{ArtDsg}}_3 = 0$, `MINIMAL_OBS`
- $y^{\mathrm{ArtDsg}}_3 = 1$, `STANDARD_OBS`
- $y^{\mathrm{ArtDsg}}_3 = 2$, `ADVANCED_OBS`

#### 2.1.2.4 $y^{\mathrm{ArtDsg}}_4$ (PERFORMANCE_OBS) [LATER]
It has three possible values:

- $y^{\mathrm{ArtDsg}}_4 = 0$, `LOW_OBS`
- $y^{\mathrm{ArtDsg}}_4 = 1$, `MEDIUM_OBS`
- $y^{\mathrm{ArtDsg}}_4 = 2$, `HIGH_OBS`

#### 2.1.2.5 $y^{\mathrm{UsrDsg}}_1$ (FEEDBACK_OBS) [LATER]
It has two possible values:

- $y^{\mathrm{UsrDsg}}_1 = 0$, `POSITIVE_OBS`
- $y^{\mathrm{UsrDsg}}_1 = 1$, `NEGATIVE_OBS`

### 2.1.3 Control factors
The action control factors to the Designer's environment (outputs from the Designer agent) are:

- $a^{\mathrm{DsgArt}}_1$ (NULL)
- $a^{\mathrm{DsgArt}}_2$ (ADJUST_UI_FEATURES_ACTION)

#### 2.1.3.1 $a^{\mathrm{DsgArt}}_1$ (NULL)
It has one possible value:

- $a^{\mathrm{DsgArt}}_1 = 0$, `NULL_ACT`

#### 2.1.3.2 $a^{\mathrm{DsgArt}}_2$ (ADJUST_UI_FEATURES_ACTION)
It has three possible values:

- $a^{\mathrm{DsgArt}}_1 = 0$, `NO_CHANGE_ACT`
- $a^{\mathrm{DsgArt}}_2 = 1$, `CHANGE_COLOR_THEME_ACT`
- $a^{\mathrm{DsgArt}}_3 = 2$, `CHANGE_TEXT_SIZE_ACT`

```
_labDsg = { ## labels for Designer
    "a": {
        "aᴰˢᵍᴬʳᵗ₁": [ ## "NULL"
            "NULL_ACT", 
        ],
        "aᴰˢᵍᴬʳᵗ₂": [ ## "ADJUST_UI_FEATURES_ACTION"
            "NO_CHANGE_ACT", 
            "CHANGE_COLOR_THEME_ACT", 
            "CHANGE_TEXT_SIZE_ACT"
        ],
    },
    "s": {
        "sᴬʳᵗᴰˢᵍ₁": [ ## "EYE_TRACKING"
            "FOCUSED", 
            "SCATTERED",
        ],
        "sᴬʳᵗᴰˢᵍ₂": [ ## "TIME_ON_TASK"
            "SHORT", 
            "MEDIUM", 
            "LONG"
        ],
        ## LATER:
        # "sᴬʳᵗᴰˢᵍ₃": [ ## "UI_FEATURES"
        #     "MINIMAL", 
        #     "STANDARD", 
        #     "ADVANCED"
        # ],
        # "sᴬʳᵗᴰˢᵍ₄": [ ## "PERFORMANCE"
        #     "LOW", 
        #     "MEDIUM", 
        #     "HIGH"
        # ],
        # "sᵁˢʳᴰˢᵍ₁": [ ## "FEEDBACK"
        #     "POSITIVE", 
        #     "NEGATIVE"
        # ],
    },
    "s̆": {
        "s̆ᴬʳᵗᴰˢᵍ₁": [ ## "EYE_TRACKING"
            "FOCUSED", 
            "SCATTERED",
        ],
        "s̆ᴬʳᵗᴰˢᵍ₂": [ ## "TIME_ON_TASK"
            "SHORT", 
            "MEDIUM", 
            "LONG"
        ],
    },    
    "y": {
        "yᴬʳᵗᴰˢᵍ₁": [ ## "EYE_TRACKING_OBS"
            "FOCUSED_OBS",
            "SCATTERED_OBS",
            "NEUTRAL_OBS"
        ],
        "yᴬʳᵗᴰˢᵍ₂": [ ## "TIME_ON_TASK_OBS"
            "SHORT_OBS",
            "MEDIUM_OBS",
            "LONG_OBS"
        ],
        "yᴬʳᵗᴰˢᵍ₃": [ ## "UI_FEATURES_OBS"
            "MINIMAL_OBS",
            "STANDARD_OBS",
            "ADVANCED_OBS"
        ],
        ## LATER:
        # "yᴬʳᵗᴰˢᵍ₄": [ ## "PERFORMANCE_OBS"
        #     "LOW_OBS",
        #     "MEDIUM_OBS",
        #     "HIGH_OBS"
        # ],
        # "yᵁˢʳᴰˢᵍ₁": [ ## "FEEDBACK_OBS"
        #     "POSITIVE_OBS",
        #     "NEGATIVE_OBS"
        # ]
    },
}
_yArtDsg_car,_yArtDsg_num, _sArtDsg_car,_sArtDsg_num, _aDsgArt_car,_aDsgArt_num = get_model_dimensions_from_labels(_labDsg)
_yArtDsg_car,_yArtDsg_num, _sArtDsg_car,_sArtDsg_num, _aDsgArt_car,_aDsgArt_num
```
```
_aDsgArt_fac_names = list(_labDsg['a'].keys()); print(f'{_aDsgArt_fac_names=}') ## control factor names
_sArtDsg_fac_names = list(_labDsg['s'].keys()); print(f'{_sArtDsg_fac_names=}') ## state factor names
_s̆ArtDsg_fac_names = list(_labDsg['s̆'].keys()); print(f'{_s̆ArtDsg_fac_names=}') ## state factor names
_yArtDsg_mod_names = list(_labDsg['y'].keys()); print(f'{_yArtDsg_mod_names=}') ## observation modality names
```

### 2.1.4 Observation likelihood matrix, $\mathbf A$
We now setup the observation likelihood matrix which is the first main component of generative model.

_Aᴬʳᵗᴰˢᵍ = utils.obj_array_zeros([[y_car] + _sArtDsg_car for y_car in _yArtDsg_car])
print(f'{len(_Aᴬʳᵗᴰˢᵍ)=}')
_Aᴬʳᵗᴰˢᵍ

#### 2.1.4.1 Observation modality $y^{\mathrm{ArtDsg}}_1$ (EYE_TRACKING_OBS)

We setup this modality's likelihood mapping, corresponding to how $y^{\mathrm{ArtDsg}}_1$ (EYE_TRACKING_OBS) is related to hidden states.

```
_Aᴬʳᵗᴰˢᵍ[0][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₁'].index('NEUTRAL_OBS'), 
    :, 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('SHORT')
] = 1.0
_Aᴬʳᵗᴰˢᵍ[0][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₁'].index('NEUTRAL_OBS'), 
    :, 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('MEDIUM')
] = 1.0
_Aᴬʳᵗᴰˢᵍ[0][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₁'].index('FOCUSED_OBS'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('FOCUSED'), 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('LONG')
] = 0.8
_Aᴬʳᵗᴰˢᵍ[0][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₁'].index('SCATTERED_OBS'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('FOCUSED'), 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('LONG')
] = 0.2

_Aᴬʳᵗᴰˢᵍ[0][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₁'].index('SCATTERED_OBS'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('SCATTERED'), 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('LONG')
] = 0.8
_Aᴬʳᵗᴰˢᵍ[0][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₁'].index('FOCUSED_OBS'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('SCATTERED'), 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('LONG')
] = 0.2

_Aᴬʳᵗᴰˢᵍ[0]
```

#### 2.1.4.2 Observation modality $y^{\mathrm{ArtDsg}}_2$ (TIME_ON_TASK_OBS)

We setup this modality's likelihood mapping, corresponding to how $y^{\mathrm{ArtDsg}}_2$ (TIME_ON_TASK_OBS) is related to hidden states.

```
_Aᴬʳᵗᴰˢᵍ[1][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₂'].index('LONG_OBS'), 
    :, 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('SHORT')
] = 1.0

_Aᴬʳᵗᴰˢᵍ[1][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₂'].index('LONG_OBS'), 
    :, 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('LONG')
] = 1.0

_FOCUSED_MAPPING_DSG = softmax(np.array([1.0, 0]))
_SCATTERED_MAPPING_DSG = softmax(np.array([0.0, 1.0]))

_Aᴬʳᵗᴰˢᵍ[1][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₂'].index('SHORT_OBS'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('FOCUSED'), 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('MEDIUM')
] = _FOCUSED_MAPPING_DSG[0]
_Aᴬʳᵗᴰˢᵍ[1][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₂'].index('MEDIUM_OBS'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('FOCUSED'), 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('MEDIUM')
] = _FOCUSED_MAPPING_DSG[1]

_Aᴬʳᵗᴰˢᵍ[1][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₂'].index('SHORT_OBS'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('SCATTERED'), 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('MEDIUM')
] = _SCATTERED_MAPPING_DSG[0]
_Aᴬʳᵗᴰˢᵍ[1][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₂'].index('MEDIUM_OBS'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('SCATTERED'), 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('MEDIUM')
] = _SCATTERED_MAPPING_DSG[1]

_Aᴬʳᵗᴰˢᵍ[1]
```

#### 2.1.4.3 Observation modality $y^{\mathrm{ArtDsg}}_3$ (UI_FEATURES_OBS)

We setup this modality's likelihood mapping, corresponding to how $y^{\mathrm{ArtDsg}}_3$ (UI_FEATURES_OBS) is related to hidden states.

```
_Aᴬʳᵗᴰˢᵍ[2][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₃'].index('MINIMAL_OBS'), 
    :, 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('SHORT')
] = 1.0
_Aᴬʳᵗᴰˢᵍ[2][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₃'].index('STANDARD_OBS'), 
    :, 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('MEDIUM')
] = 1.0
_Aᴬʳᵗᴰˢᵍ[2][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₃'].index('ADVANCED_OBS'), 
    :, 
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('LONG')
] = 1.0

_Aᴬʳᵗᴰˢᵍ[2]
```
```
print(f'=== _sArtDsg_car:\n{_sArtDsg_car}')
print(f'=== _yArtDsg_car:\n{_yArtDsg_car}')
_Aᴬʳᵗᴰˢᵍ
```
```
array([array([[[0. , 0. , 0.8],
               [0. , 0. , 0.2]],

              [[0. , 0. , 0.2],
               [0. , 0. , 0.8]],

              [[1. , 1. , 0. ],
               [1. , 1. , 0. ]]]),
       array([[[0.        , 0.73105858, 0.        ],
               [0.        , 0.26894142, 0.        ]],

              [[0.        , 0.26894142, 0.        ],
               [0.        , 0.73105858, 0.        ]],

              [[1.        , 0.        , 1.        ],
               [1.        , 0.        , 1.        ]]]),
       array([[[1., 0., 0.],
               [1., 0., 0.]],

              [[0., 1., 0.],
               [0., 1., 0.]],

              [[0., 0., 1.],
               [0., 0., 1.]]])], dtype=object)
```

### 2.1.5 State transition matrix, $\mathbf B$

```
_control_fac_idx_Dsg = [1] ## used in Agent constructor
_Bᴬʳᵗᴰˢᵍ = utils.obj_array(_sArtDsg_num); print(f'{_sArtDsg_num=}')
print(f'{len(_Bᴬʳᵗᴰˢᵍ)=}')
_Bᴬʳᵗᴰˢᵍ
```

#### 2.1.5.1 Control factor $a^{\mathrm{DsgArt}}_1$ (NULL)

We setup this factor's mapping, corresponding to how $a^{\mathrm{DsgArt}}_1$ (NULL) is related to hidden states.

_Bᴬʳᵗᴰˢᵍ[0] = np.zeros((_sArtDsg_car[0], _sArtDsg_car[0], _aDsgArt_car[0])); print(f'{_sArtDsg_car[0]=}, {_sArtDsg_car[0]=}, {_aDsgArt_car[0]=}')
_Bᴬʳᵗᴰˢᵍ[0]

```
_p_stochDsg = 0.0
## we cannot influence factor zero, set up the 'default' stationary dynamics - 
## one state just maps to itself at the next timestep with very high probability, 
## by default. So this means the EYE_TRACKING state can change from one to another with 
## some low probability (p_stoch)
_Bᴬʳᵗᴰˢᵍ[0][
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('FOCUSED'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('FOCUSED'), 
    _labDsg['a']['aᴰˢᵍᴬʳᵗ₁'].index('NULL_ACT')
] = 1.0 - _p_stochDsg
_Bᴬʳᵗᴰˢᵍ[0][
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('SCATTERED'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('FOCUSED'), 
    _labDsg['a']['aᴰˢᵍᴬʳᵗ₁'].index('NULL_ACT')
] = _p_stochDsg

_Bᴬʳᵗᴰˢᵍ[0][
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('SCATTERED'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('SCATTERED'), 
    _labDsg['a']['aᴰˢᵍᴬʳᵗ₁'].index('NULL_ACT')
] = 1.0 - _p_stochDsg
_Bᴬʳᵗᴰˢᵍ[0][
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('FOCUSED'),
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₁'].index('SCATTERED'), 
    _labDsg['a']['aᴰˢᵍᴬʳᵗ₁'].index('NULL_ACT')
] = _p_stochDsg

_Bᴬʳᵗᴰˢᵍ[0]
```

#### 2.1.5.2 Control factor $a^{\mathrm{DsgArt}}_2$ (ADJUST_UI_FEATURES_ACTION)

We setup this factor's mapping, corresponding to how $a^{\mathrm{DsgArt}}_2$ (ADJUST_UI_FEATURES_ACTION) is related to hidden states.

```
_Bᴬʳᵗᴰˢᵍ[1] = np.zeros((_sArtDsg_car[1], _sArtDsg_car[1], _aDsgArt_car[1]))
_Bᴬʳᵗᴰˢᵍ[1][
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('SHORT'), 
    :, 
    _labDsg['a']['aᴰˢᵍᴬʳᵗ₂'].index('NO_CHANGE_ACT')
] = 1.0
_Bᴬʳᵗᴰˢᵍ[1][
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('MEDIUM'), 
    :, 
    _labDsg['a']['aᴰˢᵍᴬʳᵗ₂'].index('CHANGE_COLOR_THEME_ACT')
] = 1.0
_Bᴬʳᵗᴰˢᵍ[1][
    _labDsg['s']['sᴬʳᵗᴰˢᵍ₂'].index('LONG'), 
    :, 
    _labDsg['a']['aᴰˢᵍᴬʳᵗ₂'].index('CHANGE_TEXT_SIZE_ACT')
] = 1.0

_Bᴬʳᵗᴰˢᵍ[1]
```

```
print(f'=== _aDsgArt_car:\n{_aDsgArt_car}')
print(f'=== _sArtDsg_car:\n{_sArtDsg_car}')
_Bᴬʳᵗᴰˢᵍ
```
array([array([[[1.],
               [0.]],

              [[0.],
               [1.]]]), array([[[1., 0., 0.],
                                [1., 0., 0.],
                                [1., 0., 0.]],

                               [[0., 1., 0.],
                                [0., 1., 0.],
                                [0., 1., 0.]],

                               [[0., 0., 1.],
                                [0., 0., 1.],
                                [0., 0., 1.]]])], dtype=object)

### 2.1.6 Prior preferences vector, $\mathbf C$

Now we parameterise the C vector, or the prior beliefs about observations. This will be used in the expression of the prior over actions, which is technically a softmax function of the negative expected free energy of each action. It is the equivalent of the exponentiated reward function in reinforcement learning treatments.

```
_Cᴬʳᵗᴰˢᵍ = utils.obj_array_zeros([y_car for y_car in _yArtDsg_car])
_Cᴬʳᵗᴰˢᵍ
```
```
_Cᴬʳᵗᴰˢᵍ[1][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₂'].index('SHORT_OBS'),
] = 1.0
_Cᴬʳᵗᴰˢᵍ[1][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₂'].index('MEDIUM_OBS'),
] = -1.0
_Cᴬʳᵗᴰˢᵍ[1][
    _labDsg['y']['yᴬʳᵗᴰˢᵍ₂'].index('LONG_OBS'),
] = 0.0

_Cᴬʳᵗᴰˢᵍ[1]
```

### 2.1.7 Initialise an instance of the `Agent()` class:

```
_agtDsg = Agent(
    A=_Aᴬʳᵗᴰˢᵍ, 
    B=_Bᴬʳᵗᴰˢᵍ, 
    C=_Cᴬʳᵗᴰˢᵍ, 
    control_fac_idx=_control_fac_idx_Dsg
)
_agtDsg
```

## 2.2 Designer environment
This is the Designer agent's *generative process* for its environment.

It is important to note that the generative process doesn't have to be described by A and B matrices - it can just be the arbitrary 'rules of the game' that you 'write in' as a modeller. But here we just use the same transition/likelihood matrices to make the sampling process straightforward.

```
## observation/transition matrices characterising the generative process

## currently only true values of Artifact
## should be _Ăᴰˢᵍ to include true values of the complete Designer environment
_Ăᴬʳᵗᴰˢᵍ = copy.deepcopy(_Aᴬʳᵗᴰˢᵍ)

## True next-state may be calculated without a B matrix
## currently only true values of Artifact
## should be _B̆ᴰˢᵍ to include true values of the complete Designer environment
_B̆ᴬʳᵗᴰˢᵍ = copy.deepcopy(_Bᴬʳᵗᴰˢᵍ)
```