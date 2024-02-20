import numpy as np
import tensorflow as tf

qw=137
vw=1000
weight=np.ones(28)
weight[1]*=qw*vw
weight[3]*=qw*vw
weight[4]*=qw*qw
weight[5]*=vw
weight[6]*=vw
weight[7]*=qw
weight[8]*=qw
weight[9]*=qw
weight[10]*=vw
weight[11]*=vw*qw
weight[12]*=qw*vw
weight[13]*=qw*vw*qw
weight[1+14]*=qw*vw
weight[3+14]*=qw*vw
weight[4+14]*=qw*qw
weight[5+14]*=vw
weight[6+14]*=vw
weight[7+14]*=qw
weight[8+14]*=qw
weight[9+14]*=qw
weight[10+14]*=vw
weight[11+14]*=vw*qw
weight[12+14]*=qw*vw
weight[13+14]*=qw*vw*qw



def r(v, nQ):
    s = np.exp(v[:,0])
    return s[:,np.newaxis] * np.exp(v[:,1:nQ + 1])

def inference(material,c1s,c3s,c4s,c5s,c6s,c7s,c8s,c9s,c10s,c11s,c12s,c13s,c14s,c15s,
            c1l,c3l,c4l,c5l,c6l,c7l,c8l,c9l,c10l,c11l,c12l,c13l,c14l,c15l,m_x):

    if np.max(m_x)>1000:
        raise ValueError('DM mass larger than 1GeV, network operating outside training range')
    if np.min(m_x)<3 and material=='Ge':
        raise ValueError('DM mass smaller than 3.1 MeV for Germanium, network operating outside training range')
    if np.min(m_x)<4 and material=='Si':
        raise ValueError('DM mass smaller than 4.0 MeV for Silicon, network operating outside training range')

    cvec=np.zeros((np.size(c1s),28))
    cvec[:, 0] = c1s / weight[0]
    cvec[:, 1] = c3s / weight[1]
    cvec[:, 2] = c4s / weight[2]
    cvec[:, 3] = c5s / weight[3]
    cvec[:, 4] = c6s / weight[4]
    cvec[:, 5] = c7s / weight[5]
    cvec[:, 6] = c8s / weight[6]
    cvec[:, 7] = c9s / weight[7]
    cvec[:, 8] = c10s / weight[8]
    cvec[:, 9] = c11s / weight[9]
    cvec[:, 10] = c12s / weight[10]
    cvec[:, 11] = c13s / weight[11]
    cvec[:, 12] = c14s / weight[12]
    cvec[:, 13] = c15s / weight[13]
    cvec[:, 14] = c1l / weight[14]
    cvec[:, 15] = c3l / weight[15]
    cvec[:, 16] = c4l / weight[16]
    cvec[:, 17] = c5l / weight[17]
    cvec[:, 18] = c6l / weight[18]
    cvec[:, 19] = c7l / weight[19]
    cvec[:, 20] = c8l / weight[20]
    cvec[:, 21] = c9l / weight[21]
    cvec[:, 22] = c10l / weight[22]
    cvec[:, 23] = c11l / weight[23]
    cvec[:, 24] = c12l / weight[24]
    cvec[:, 25] = c13l / weight[25]
    cvec[:, 26] = c14l / weight[26]
    cvec[:, 27] = c15l / weight[27]
    cvecmax = np.max(abs(cvec)) * 1.1
    if np.size(c1s)==1:
        params = np.zeros((2, 29))
        params[0, 0:28] = cvec / cvecmax
        if material=='Ge':
                params[0, 28] = -2 * np.log(m_x / 1E3)/np.log(3.0 / 1E3)  + 1
        elif material=='Si':
                params[0, 28] = -2 * np.log(m_x / 1E3) / np.log(4.0 / 1E3)  + 1
        else:
                raise ValueError('Invalid material, must be Si or Ge')
        params.reshape((2, 29, 1))
    else:
        m_x=np.asarray(m_x)
        params = np.zeros((len(c1s), 29))
       	params[:, 0:28] = cvec / cvecmax
        if material=='Ge':
                params[:, 28] = -2 * np.log(m_x / 1E3)/np.log(3.0 / 1E3)  + 1
        elif material=='Si':
                params[:, 28] = -2 * np.log(m_x / 1E3) / np.log(4.0 / 1E3)  + 1
        else:
                raise ValueError('Invalid material, must be Si or Ge')
        params.reshape((len(c1s), 29, 1))

    model = tf.keras.models.load_model(material+'.h5')
    Qp = model.predict(params)
    if np.size(c1s)==1:
        return r(Qp,4)[0,:]*cvecmax**2

    else:
        return r(Qp, 4) * cvecmax ** 2
