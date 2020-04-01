# NCE_GAN

Noise-contrastive estimation is a relatively new approach (Gutmann and Hyvärinen,
2010, 2012) to handle missing normalising (multiplicative) constants in complex densities. It is notably used to to modify the  softmax loss function of neural language models (which is costly due to the large vocabulary ) in a less costly loss function.  The
missing constant is integrated within the parameter space and estimated by logistic regression
(Geyer, 1994). Since this fairly effective method depends on a pseudo-sample simulated from
an instrumental distribution pn, this distribution could be optimised and the concept of generative adversarial network (GAN, Goodfellow et al., 2014) seems appropriate for handling this
optimisation as a supervised learning problem


Gutmann, M. U. and Hyvärinen, A. (2012). Noise-contrastive estimation of unnormalized
statistical models, with applications to natural image statistics. J. Mach. Learn. Res., 13,
307–361. http://dl.acm.org/citation.cfm?id=2188385.2188396.


