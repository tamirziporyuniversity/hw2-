import torch
import torch.nn as nn
from torch.autograd import Variable

class OurModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(OurModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)  # In the terms of the handout, here d = D_h
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_, hidden):
        # General instructions:
        # Pass the embedded input through the GRU and use the output layer to get the next character distribution.
        # return that distribution and the next hidden state.
        # You may need to play around with the dimensions a bit until you get it right. Dimension-induced frustration is good for you!
        # -------------------------
        # YOUR CODE HERE
        #print("here0")
        #print(input_.shape)
        print(input_)
        embedding_vectors = self.embedding(input_)
        print(embedding_vectors)
        print(embedding_vectors.shape)
        embedding_vectors = embedding_vectors.unsqueeze(1).unsqueeze(1).permute(2, 1, 0)
        #out = embedding_vectors
        # print(self.input_size)
        # print(self.hidden_size)
        # print(embedding_vectors.shape)
        # print(hidden.shape)
       
        out = embedding_vectors
        for i in range(self.num_layers):
          out, hidden = self.gru(out, hidden)
        #print("here2")
        output = self.output_layer(out.squeeze(1))

        

        #raise NotImplementedError()
        # -------------------------
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
    


    """
    [time elapsed: 0m 32s  ;  epochs: 100 (5.0%)  ;  loss: 2.22]
What 'od fove his anden prod
And mand fnor dour neathon toat to owrigce, sim in for, wiiy ourwides.
Agt
Fer wiick with mave of icet Wraing.

A3 I thed come hing the thobecherend ttoceng;
Yow, loneern: t 

[time elapsed: 1m 5s  ;  epochs: 200 (10.0%)  ;  loss: 1.931]
Whay to have care and ave you hereis and shetater withy meor with here the pear thery sunigh the by ke we and havence will gratcles, bebour to Jurt you well!
Whow hoo thell prom,
A a now the now grare w 

[time elapsed: 1m 39s  ;  epochs: 300 (15.0%)  ;  loss: 1.926]
Whall welly that I bet coned in not if for lord;
And make! and sto the prathels,
And by fith not brese be brast the some betherting bath brore
Wit hard and words keat arave amr now ongh, theer cody kith 

[time elapsed: 2m 12s  ;  epochs: 400 (20.0%)  ;  loss: 1.966]
Whall hall your the dows;
He now you shave have ir hell whould mone the dow will encour heep
Why the make and goth sir, he brour your there bind,
To prown the call have cave you?

CHOMIO:
Por so me have 

[time elapsed: 2m 46s  ;  epochs: 500 (25.0%)  ;  loss: 1.827]
When:
I tall be heous pooly to and say, you shall your'd fly wa, dather,---
Prmam!

DICKI Hard I go me a god Parl
With seder hond, here makest thou gaine madans to acheet's the make that in,
And stay st 

[time elapsed: 3m 19s  ;  epochs: 600 (30.0%)  ;  loss: 1.945]
Which and thou as
O thou a from the canus.

KING RICHARD II:
Fity than prove tence there of what me;
A our biven the do to that wating to lis preise
The proves on pilliver whirs thraves sheel, the me's  

[time elapsed: 3m 52s  ;  epochs: 700 (35.0%)  ;  loss: 1.613]
Wh/
Mast thou cay, make ure, with ince she sone
This I to strane is and rewing.

QUEEN MERCIO:
Derame a hall partion, to for shald and so usted thy to sir.
Well for ham for weats at thou lated of low, s 

[time elapsed: 4m 26s  ;  epochs: 800 (40.0%)  ;  loss: 1.732]
When and them, the curling
Woo heaven for you sous that the I how; why why,
That of shave pall thene, have lodry; the and son and
Thouse thou sent the call sons hear.
That I have thou more take intion t 

[time elapsed: 4m 59s  ;  epochs: 900 (45.0%)  ;  loss: 1.78]
Whand, fillowans else well Vever,
To he with bid 'entant may it:
I masty a quein:
And with hear hom, and chost,
Will uphe sonatment, too as with this besece,
I'll be come I your for his for the us call  

[time elapsed: 5m 32s  ;  epochs: 1000 (50.0%)  ;  loss: 1.762]
Which it me this wap.

PENTIO:
Which how die, all this the vegthing nockes,
Which do rood to reg quit of him, thou hast
To then we a my be thou came these him Biar,
Trartuest Montrew under woo marrt
The 

[time elapsed: 6m 6s  ;  epochs: 1100 (55.00000000000001%)  ;  loss: 1.648]
Which in the great a condeds
I parven will withers your my sapiour
The maging a wit. He tor affor store;
That be a dornatear, from thou news ig may,
I pray for you do the intreat,
The counders why rest  

[time elapsed: 6m 38s  ;  epochs: 1200 (60.0%)  ;  loss: 1.62]
Whand daught
Whose wil, sir; our belive loves not pailenter it be a
gent thou ishes the golder of not yet our the prither
We from fixked the pation it of hath for he welcomorus,
Oury and me make deceith 

[time elapsed: 7m 12s  ;  epochs: 1300 (65.0%)  ;  loss: 1.783]
Why seet is in he man.

CLAUDIO:
Nay, but these his mark with the thath of he should seep it.

GREMIO:
Come fanspiled of come calling like him!

LUCENTIO:
Ay, it the woul the know; Say have laveit.

VIN 

[time elapsed: 7m 45s  ;  epochs: 1400 (70.0%)  ;  loss: 1.622]
Whand, a be so his be badow!

KATHARINA:
When norly he down this upon'done me the it.

KING RICHARD II:
Greach the me, os to lords!

MIRAND:
That I the preace,
And didy's ald thy mist Marge!

HORMIO:
Wh 

[time elapsed: 8m 18s  ;  epochs: 1500 (75.0%)  ;  loss: 1.784]
Why!
Sall of my life with to the fort a bencless
And with exeep breath untess the kneed thean
For betch entoan thy there
we and thee send me seen, fate too not stone
There fin that feach too breath'm re 

[time elapsed: 8m 52s  ;  epochs: 1600 (80.0%)  ;  loss: 1.82]
Why with not with the will so
we deep the suithing the quiet thy my greator's must act's not?

KING RICHARD III:

KING RICHARD III:
I will I'll them greep will heart the lage and unthe reacher,
You he s 

[time elapsed: 9m 24s  ;  epochs: 1700 (85.0%)  ;  loss: 1.693]
Why: I known the readers, midest the everest.

MAGANUS:
Hoose the my destrume, blent,
It shall nobis! shall I haste the soepy here and and when in Pear
And for the aplede for this all this you not righa 

[time elapsed: 9m 57s  ;  epochs: 1800 (90.0%)  ;  loss: 1.719]
Why him hadge,
The was, grawime the have bod, the bost by from.

CLARENCE:
That we came at the wout to my lords lord:
The with this shall but a arm'd that one his the dear yet there but too:
Good of mos 

[time elapsed: 10m 31s  ;  epochs: 1900 (95.0%)  ;  loss: 1.462]
Whany.

JULIET:
And have and my long say, file!
O lanted and heset unto standt
Trange with me and means
My made as the know hath hear?

BAPTISTA:
How we as lands masten I do we wo.

CAPULETS:
It have so 

[time elapsed: 11m 4s  ;  epochs: 2000 (100.0%)  ;  loss: 1.752]
Whating so took;
And, devindow the still be this is my he poiling,
And come: 'tis to besent if them?

CORIOLANUS:
Ay, be sting the have are mother sitizen, my heart.

Thrwicly Citizng:
Not com so, so th 

    """