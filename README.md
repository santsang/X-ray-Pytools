# X-ray-Pytools

I post certain [scripts](https://github.com/santsang/X-ray-Pytools/tree/script) I wrote for a _fellowship.ai_ 3-months project.  It was only a computational success, but not modelling success.  I only knew very few python commands when I started the project.  I did not even know what _channel_ is.  Eventually I built the skills of reading the codes quickly.  

The purpose of the research is to apply Articifical Intelligence (AI) approach to classify X-ray images of Spinal Implants from eight manufacturers.  The coding of outcome/classification was not certain, as some images were verified by radiologist (sometimes with annotations) and some from social media.  I eliminated those seemingly gathered from social media.  It left me with 452 images demonstrating diverse qualities and imbalance classes.  One can expect that I got poor results, although I tried literally almost all possible techniques, including transfer learning.  I did not try gridsearch techniques.  I doubt that gridsearch is a magic bullet.  We need to have a reasonable good model in order to obtain a desirable result.  The best I could get was about 29% ish by implementing a VGG-19 and 56% ish by building a Siamese Network.  Various authors claimed having successes with VGG-based models.  In most occasions, the models did not learn.  I got similar outcomes with models with various filter sizes, but still could not emerge a reasonable model.

I bet the scripts more or less cover the full-stack service - editing, cleaning.., and modelling.  They may be of use to others.  I did not apply subclass to build a model.  It is easier to avoid mistakes by using Sequential or Functional APIs.

I may mention that I acquired most of the skills from Dr. Adrian Rosebrock's [site](https://pyimagesearch.com) and also Dr. Jason Brownlee's [site](https://machinelearningmastery.com/).
