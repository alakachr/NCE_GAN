# NCE_GAN

Noise-contrastive estimation is a relatively new approach (Gutmann and Hyvärinen,
2010, 2012) to handle missing normalising (multiplicative) constants in complex densities. The
missing constant is integrated within the parameter space and estimated by logistic regression
(Geyer, 1994). Since this fairly effective method depends on a pseudo-sample simulated from
an instrumental distribution pn, this distribution could be optimised and the concept of generative adversarial network (GAN, Goodfellow et al., 2014) seems appropriate for handling this
optimisation as a supervised learning problem
