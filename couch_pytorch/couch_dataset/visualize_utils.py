import numpy as np

xsens_to_goal_transformation = {'session1': {'chair1': {'height': 0., 'angle': 0.},
                                             'chair2': {'height': 0., 'angle': -1.7}},
                                'session2': {'chair1': {'height': 0., 'angle': 0.},
                                             'chair2': {'height': 0., 'angle': -1.7},
                                             'chair3': {'height': 0., 'angle': 0.1},
                                             'sofa': {'height': -0.1, 'angle': -0.5}},
                                'session3': {'chair1': {'height': 0., 'angle': 0.},
                                             'chair2': {'height': 0., 'angle': -1.7},
                                             'sofa': {'height': -0.1, 'angle': -0.5}},
                                'session4': {'chair1': {'height': 0., 'angle': 0.},
                                             'chair2': {'height': 0., 'angle': -1.7},
                                             'sofa': {'height': -0.1, 'angle': -0.5}},
                                'session5': {'chair1': {'height': 0., 'angle': 0.},
                                             'chair2': {'height': 0., 'angle': -1.7},
                                             'sofa': {'height': -0.1, 'angle': -0.5}},
                                }

all_objects_static_params = {
    'session1': {'subject1': {'chair1': {'trans': np.array([-0.24597032, 0.72961587, 2.5225177], dtype=np.float32),
                                         'angle': np.array([3.0902104, 0.01569591, -0.16500677], dtype=np.float32)},
                              'chair2': {'trans': np.array([0.05, 0.7656641, 2.25], dtype=np.float32),
                                         # 'trans2': np.array([0.04192, 0.7656641, 2.287489], dtype=np.float32),
                                         'angle': np.array([-2.2103207, 0.31618354, -2.408768], dtype=np.float32)}}},

    'session2': {'subject2': {'chair1': {'trans': np.array([0.06486525, 0.74251527, 2.1792488], dtype=np.float32),
                                         'angle': np.array([3.1171536, 0.01731278, -0.05689536], dtype=np.float32)},
                              'chair2': {'trans': np.array([0.0175335, 0.768191, 2.2981927], dtype=np.float32),
                                         'angle': np.array([-1.8578218, 0.32985106, -2.697772], dtype=np.float32)}
                              },
                 'subject1': {'chair3': {'trans': np.array([-0.02871853, 0.5939967, 2.42943536], dtype=np.float32),
                                         'angle': np.array([3.1002586, 0.03046918, -0.21584417], dtype=np.float32)},
                              'sofa': {'trans': np.array([-0.12205392, 0.7249135, 2.3961742], dtype=np.float32),
                                       'angle': np.array([3.0558898, 0.0364731, 0.67788607], dtype=np.float32)}}
                 },
    'session3': {'subject3': {'chair1': {'trans': np.array([-0.12994675, 0.830161, 2.285051], dtype=np.float32),
                                         'angle': np.array([3.0340044, 0.09537666, 0.4938207], dtype=np.float32)},
                              'chair2': {'trans': np.array([-0.08169547, 0.8636101, 2.1873912], dtype=np.float32),
                                         'angle': np.array([-2.1166565, 0.22181112, -2.5336392], dtype=np.float32)},
                              'sofa': {'trans': np.array([-0.0887935, 0.85072395, 2.1847558], dtype=np.float32),
                                       'angle': np.array([3.0172853, 0.1222362, 0.7305713], dtype=np.float32)}
                              }},
    'session4': {'subject1': {'chair1': {'trans': np.array([0.02866027, 0.75258945, 2.447922], dtype=np.float32),
                                         'angle': np.array([3.1296582, 0.03992375, -0.03911312], dtype=np.float32)},
                              'chair2': {'trans': np.array([0.129601, 0.7614854, 2.568968], dtype=np.float32),
                                         'angle': np.array([-2.2103207, 0.35618354, -2.408768], dtype=np.float32)},
                              'sofa': {'trans': np.array([0.00601, 0.7314854, 2.558968], dtype=np.float32),
                                       'angle': np.array([3.0902104, 0.0569591, 0.65677], dtype=np.float32)}},
                 'subject4': {'chair1': {'trans': np.array([0.00293461, 0.7008212, 2.5903268], dtype=np.float32),
                                         'angle': np.array([3.1691267, 0.0394704, -0.05680508], dtype=np.float32)},
                              'chair2': {'trans': np.array([0.08887382, 0.7553532, 2.5227878], dtype=np.float32),
                                         'angle': np.array([-2.1239002, 0.31290352, -2.4995847], dtype=np.float32)},
                              'sofa': {'trans': np.array([-0.167768, 0.7363245, 2.4565799], dtype=np.float32),
                                       'angle': np.array([3.04978, 0.00527559, 0.721556], dtype=np.float32)}
                              }
                 },
    'session5': {'subject5': {'chair1': {'trans': np.array([0.04007136, 0.7171893, 2.2639678], dtype=np.float32),
                                         'angle': np.array([3.1184874, 0.00598939, 0.02197747], dtype=np.float32)},
                              'chair2': {'trans': np.array([0.089501, 0.7634055, 2.3124713], dtype=np.float32),
                                         'angle': np.array([-2.1820061, 0.3217656, -2.4552486], dtype=np.float32)},
                              'sofa': {
                                  'trans': np.array([-3.0260529e-02, 7.2931560e-01, 2.5334659e+00], dtype=np.float32),
                                  'angle': np.array([3.1726938, 0.02122523, 0.78964654], dtype=np.float32)}
                              },
                 'subject6': {'chair1': {'trans': np.array([0.02132054, 0.724614, 2.4900146], dtype=np.float32),
                                         'angle': np.array([3.1362126, 0.04590815, 0.03559541], dtype=np.float32)},
                              'chair2': {'trans': np.array([-0.10246087, 0.75735104, 2.442401], dtype=np.float32),
                                         'angle': np.array([-1.9863132, 0.32998747, -2.6006892], dtype=np.float32)},
                              'sofa': {
                                  'trans': np.array([-3.0260529e-02, 7.348901e-01, 2.3841288e+00], dtype=np.float32),
                                  'angle': np.array([3.0771573e+00, 1.9647189e-03, 7.8490094e-01], dtype=np.float32)
                                  }
                              }
                 }
}

SCAN_PATH = {
    'chair1': "objects/chair1.ply",
    'chair2': "objects/chair2.ply",
    'chair3': "objects/chair3.ply",
    'sofa': "objects/sofa.ply"
}

def floor_mesh(centre=(0, 0, 0), scale=(1.8, 1.8, 1.8)):
    import numpy as np
    from psbody.mesh import Mesh
    min_ = -scale[0] / 2
    max_ = scale[0] / 2
    vertices = np.array([
        [centre[0] + min_, 0, centre[2] + max_],
        [centre[0] + max_, 0, centre[2] + min_],
        [centre[0] + max_, 0, centre[2] + max_],
        [centre[0] + min_, 0, centre[2] + min_]])
    faces = np.array([[0, 2, 3], [0, 1, 3], [1, 3, 2]])
    return Mesh(vertices, faces, vc='snow')
