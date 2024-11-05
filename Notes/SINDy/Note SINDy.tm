<TeXmacs|2.1.4>

<style|generic>

<\body>
  <doc-data|<doc-title|Note SINDy>>

  SINDy, ou \PSparse Identification of Nonlinear Dynamics\Q, est une m�thode
  d'identification automatique de mod�le � partir de donn�es, reposant sur un
  probl�me de minimisation construit pour assurer autant la fid�lit� aux
  donn�es du mod�le obtenu ainsi que sa \Psparsit�\Q (mod�le parcimonieux).

  Le <hlink|mod�le original|https://arxiv.org/abs/1509.03580> de Brunton et
  al. (2015) a connu plusieurs variantes :
  <hlink|PDE-FIND|https://arxiv.org/abs/1609.06401>, <hlink|SINDy with
  control|https://www.sciencedirect.com/science/article/pii/S2405896316318298?ref=cra_js_challenge&fr=RR-1>,
  <hlink|SINDy with sparse relaxed regularized
  regression|https://arxiv.org/abs/1906.10612> et
  <with|font-series|bold|<hlink|SINDy-PI|https://arxiv.org/abs/2004.02322>>
  (implicit-SINDy), que nous d�taillons plus loin.

  <section|Mod�lisation fondamentale>

  On suppose que l'on dispose de donn�es temporelles pour une fonction
  <math|x<around*|(|t|)>> sous la forme de deux vecteurs,
  <math|<with|font-series|bold|x>> et <math|<wide|<with|font-series|bold|x>|\<dot\>>>
  de taille <math|N>. On suppose aussi qu'il existe un <math|f> tel que

  <\equation>
    <wide|x|\<dot\>><around*|(|t|)>=f<around*|(|x<around*|(|t|)>|)>,
  </equation>

  que l'on va tenter de reconstruire � partir des donn�es. On consid�re une
  biblioth�que compos�e de <math|K> fonctions candidates

  <\equation*>
    \<Theta\>=<matrix|<tformat|<table|<row|<cell|f<rsub|1> f<rsub|2>
    \<ldots\>>>>>>,
  </equation*>

  et l'on supposera que <math|f> s'�crit comme
  <with|font-shape|italic|combinason lin�aire parcimonieuse> de fonctions de
  cette biblioth�que. On construit la matrice de taille <math|N\<times\>K>
  associ�e � la biblioth�que et aux donn�es :

  <\equation*>
    <with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>=<matrix|<tformat|<table|<row|<cell|\<theta\><rsub|1><around*|(|<with|font-series|bold|x<rsub|1>>|)><space|1em>\<theta\><rsub|2><around*|(|<with|font-series|bold|x<rsub|1>>|)>
    \<ldots\>>>|<row|<cell|\<theta\><rsub|1><around*|(|<with|font-series|bold|x<rsub|2>>|)><space|1em>\<theta\><rsub|2><around*|(|<with|font-series|bold|x<rsub|2>>|)>
    \<ldots\>>>|<row|<cell|\<vdots\><space|3em>\<vdots\><space|2em>>>>>>,
  </equation*>

  et le probl�me revient maintenant � trouver un vecteur
  <math|\<xi\>\<in\>\<bbb-R\><rsup|K>> de norme
  <math|\<\|\|\>\<cdot\>\<\|\|\><rsub|0>> la plus petite possible et qui
  minimise \<\|\|\><math|<wide|<with|font-series|bold|x>|\<dot\>>> <math|->
  <math|<with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>\<xi\>>\<\|\|\>.\ 

  <paragraph|Probl�me de minimisation>

  Pr�cis�ment, le probl�me initial s'�crit donc

  <\equation*>
    <below|argmin|\<xi\>\<in\> \<bbb-R\><rsup|K>>
    \<\|\|\><wide|<with|font-series|bold|x>|\<dot\>>-<with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>\<xi\>\<\|\|\><rsub|2><space|0.6spc>+<space|0.6spc>\<lambda\>\<\|\|\>\<xi\>\<\|\|\><rsub|0>,
  </equation*>

  o� <math|\<lambda\>\<geq\> 0> es un param�tre de r�gularisation. Ce
  probl�me est n�anmoins np-dur, ainsi on pourra le relaxer en le LASSO

  <\equation*>
    <below|argmin|\<xi\>\<in\> \<bbb-R\><rsup|K>>
    \<\|\|\><wide|<with|font-series|bold|x>|\<dot\>>-<with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>\<xi\>\<\|\|\><rsub|2><space|0.6spc>+<space|0.6spc>\<lambda\>\<\|\|\>\<xi\>\<\|\|\><rsub|1>.
  </equation*>

  <paragraph|Sequential Thresholded Least Squares>

  Le LASSO peut devenir cher computationellement pour les larges datasets, et
  de plus n'est pas tr�s adapt� � la s�lection de coefficients. Une
  alternative propos�e, qui se trouve de plus �tre simple et robuste au
  bruit, est la m�thode des <with|font-shape|italic|Sequential Thresholded
  Least Squares>, o� l'on imposera la sparsit� \P� la main\Q, en effectuant
  r�cursivement une r�gression des moindres carr�s, c'est � dire en r�solvant
  <math|<below|argmin|\<xi\>\<in\> \<bbb-R\><rsup|K>>
  \<\|\|\><wide|<with|font-series|bold|x>|\<dot\>>-<with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>\<xi\>\<\|\|\><rsub|2>>,
  puis en �liminant de la libraire <math|\<Theta\>> les fonctions pour
  lesquelles le coefficient associ� est plus faible qu'un certain cut-off
  <math|\<lambda\>>. L'identification de ce param�tre peut s'effectuer par de
  la <with|font-series||validation> crois�e.

  <paragraph|Dimension>Si <math|x> n'est pas � valeurs dans <math|\<bbb-R\>>
  mais <math|\<bbb-R\><rsup|d>> avec <math|d> grand, la m�thode SINDy
  deviendra moins adapt�e et il faudra recourir � des m�thodes de r�duction
  de dimension. Pour notre probl�me, il faudra voir si c'est un souci ou non
  : �a d�pendra de comment l'on g�re les interactions.

  <paragraph|EDO � param�tres, forcing, contr�le>Il est possible d'�tendre
  SINDy de <math|<around*|(|1|)>> � des �quations � param�tres, dependantes
  du temps ou avec <hlink|for�age|https://www.sciencedirect.com/science/article/pii/S2405896316318298?ref=cra_js_challenge&fr=RR-1>
  simplement en ajoutant des fonctions dans la librairie.

  <paragraph|Une illustration>

  <\big-figure|<image|/Users/thomasborsoni/Desktop/Post-doc/Projet
  fourmis/Programmes/Notes/SINDy/SINDy_1.png|1par|||>>
    Repr�sentation sch�matique de SINDy. <with|font-shape|italic|Brunton et
    al. 2015>
  </big-figure>

  \;

  <section|PDE-FIND>

  Une extension � SINDy pour retrouver des EDP a aussi �t� propos�e dans
  <hlink|cet article|https://arxiv.org/abs/1609.06401>. L'extension est assez
  naturelle : supposons que l'on �tudie un fonction
  <math|u<around*|(|t,x|)>>, satisfaisant une EDP de la forme

  <\equation*>
    \<partial\><rsub|t>u=N<around*|(|u,\<partial\><rsub|x>u,\<partial\><rsup|2><rsub|x
    x>u,\<ldots\>,x,t|)>,
  </equation*>

  et que l'on cherche � retrouver la fonction <math|N>. La m�thode PDE-FIND
  est une variante de SINDy o� la biblioth�que <math|\<Theta\>> contiendra
  des fonctions de <math|u>, <math|\<partial\><rsub|x>u>,
  <math|\<partial\><rsup|2><rsub|x x>u>, <math|x> et <math|t>.

  <section|SINDy with sparse regularized regression>

  La m�thode SR3 propos�e dans <hlink|cet
  article|https://arxiv.org/abs/1906.10612> propose une alternative � la
  m�thode de Sequential Thresholded Least Squares (STLSQ). Elle en est une
  sorte de g�n�ralisation, bien que STLSQ ne soit pas r�ellement incluse.

  Le probl�me de minimisation �tudi� sera ici

  <\equation*>
    <below|argmin|W\<in\> \<bbb-R\><rsup|K>><space|1em><below|min|\<xi\>\<in\>
    \<bbb-R\><rsup|K>> <frac|1|2>\<\|\|\><wide|<with|font-series|bold|x>|\<dot\>>-<with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>\<xi\>\<\|\|\><rsup|2><rsub|2><space|0.6spc>
    +<space|0.6spc> <frac|1|2 \<nu\>>\<\|\|\>\<xi\>-W\<\|\|\><rsup|2><rsub|2><space|0.6spc>
    +<space|0.6spc> \<lambda\> R<around*|(|W|)>
  </equation*>

  Si l'on prend comme fonction <math|R> la norme <math|\<ell\><rsup|0>>, ce
  probl�me sera similaire � STLSQ. On peut commencer avec STLSQ, puis passer
  � SR3 si l'on est limit�.

  <section|<hlink|SINDy-PI|https://arxiv.org/abs/2004.02322>, une m�thode
  encore plus efficace>

  Le potentiel probl�me central de la m�thode SINDy est que l'on doit
  supposer que la fonction <math|f> est une
  <with|font-shape|italic|combinaison lin�aire> d'�l�ments de la biblioth�que
  <math|\<Theta\>>, ce qui restreint fortement les cas pour lesquels la
  m�thode pourrait fonctionner.

  L'id�e propos�e est alors de plut�t �tudier le probl�me dit \Pimplicite\Q

  <\equation*>
    f<around*|(|x,<wide|x|\<dot\>>|)>=0,
  </equation*>

  pour lequel on va alors tenter de reconstruire <math|f> comme
  <with|font-shape|italic|combinaison lin�aire parcimonieuse> de fonctions
  appartenant � la librairie <math|\<Theta\><around*|(|x,<wide|x|\<dot\>>|)>>.
  � noter que le cas standard se retrouve avec la fonction
  <math|f<around*|(|x,y|)>=y-g<around*|(|x|)>>.

  Dans ce cas, on consid�re alors une librairie <math|\<Theta\>> contenant
  des fonctions de <math|x> et de <math|<wide|x|\<dot\>>>, par exemple
  <math|<around*|{|1,x,x<rsup|2>,<wide|x|\<dot\>>,x<wide|x|\<dot\>>,x<rsup|2><wide|x|\<dot\>>|}>>.
  Notons que cette librairie a alors acc�s aux �quations comme
  <math|<wide|x|\<dot\>>=<frac|x|1+x>>.

  <paragraph|Implicit SINDy>Le probl�me, dans le cadre de SINDy, est alors de
  trouver le vecteur <math|\<xi\>> le plus sparse possible et tel que l'on
  ait

  <\equation*>
    <with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>,<with|font-series|bold|<wide|x|\<dot\>>>|)>
    \<xi\>\<approx\>0.
  </equation*>

  Ce probl�me est n�anmoins mal pos�, l'�quation ci-dessus �tant invariante
  par la multiplication d'un constante. Il faut donc fixer la valeur d'une
  norme de <math|\<xi\> ><math|<around*|(|par exemple sa norme 2|)>>. On
  obtient alors le probl�me de minimisation

  <\equation*>
    <below|argmin|\<xi\>><space|1em>\<\|\|\><with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>,<with|font-series|bold|<wide|x|\<dot\>>>|)>
    \<xi\>\<\|\|\><rsub|2>+\<lambda\>\<\|\|\>\<xi\>\<\|\|\><rsub|p>,<space|1em>
    <space|1em> \<\|\|\>\<xi\>\<\|\|\><rsub|2>=1,
  </equation*>

  o� en th�orie on souhaiterait <math|p=0> mais l'on mettrait en pratique
  <math|p=1>. La r�solution de ce probl�me se trouve �tre difficile, � cause
  de la contrainte, et l'on peut en fait appliquer une m�thode plus efficace
  et robuste au bruit.

  <paragraph|SINDy-PI>L'accronyme \PPI\Q vaut pour \PParallel, Implicit\Q.

  L'id�e de SINDy-PI est de <with|font-shape|italic|traiter le probl�me
  implicite> d�crit plus haut <with|font-shape|italic|avec la m�thode
  explicite> traditionnelle. Consid�rons une librairie
  <with|font-series|bold|<math|\<Theta\><around*|(|x,<wide|x|\<dot\>>|)>>>.
  Pour toute colonne <math|<with|font-series|bold|\<theta\>><rsub|j>> de
  <math|<with|font-series|bold|\<Theta\>>>, on va r�soudre le probl�me SINDy
  standard associ� � l'�quation

  <\equation*>
    <with|font-series|bold|\<theta\>><rsub|j>=<with|font-series|bold|\<Theta\>><rsup|j>
    <with|font-series||\<xi\>><rsup|j>,
  </equation*>

  o� <math|<with|font-series|bold|\<Theta\>><rsup|j>> est la librairie
  <with|font-series|bold|<math|\<Theta\>>> � laquelle on a retir� la colonne
  <math|j>. Par suite, on fait la chose suivante : si le vecteur
  <math|\<xi\><rsup|j>> s�lectionn� n'est pas sparse et/ou donne une mauvaise
  pr�diction, on rejette la fonction <math|<with|font-series|bold|\<theta\>><rsub|j>>
  et l'on recommence apr�s avoir retir� <math|<with|font-series|bold|\<theta\>><rsub|j>>
  de la librairie. D'un autre c�t�, si l'on obtient un vecteur sparse et qui
  donne une bonne pr�diction, on peut s'arr�ter.

  On peut aussi consid�rer <with|font-series||de<with|font-series||ux>>
  librairies distinctes pour les termes de gauche et de droite.

  <\big-figure|<image|/Users/thomasborsoni/Desktop/Post-doc/Projet
  fourmis/Programmes/Notes/SINDy/SINDy_2.png|1par|||>>
    Repr�sentation de SINDy-PI. <with|font-shape|italic|Kaheman et al. 2020>
  </big-figure>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|4|3>>
    <associate|auto-11|<tuple|1|3>>
    <associate|auto-12|<tuple|2|3>>
    <associate|auto-13|<tuple|2|4>>
    <associate|auto-2|<tuple|1|1>>
    <associate|auto-3|<tuple|2|1>>
    <associate|auto-4|<tuple|3|2>>
    <associate|auto-5|<tuple|4|2>>
    <associate|auto-6|<tuple|5|2>>
    <associate|auto-7|<tuple|1|2>>
    <associate|auto-8|<tuple|2|2>>
    <associate|auto-9|<tuple|3|3>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|figure>
      <tuple|normal|<\surround|<hidden-binding|<tuple>|1>|>
        Repr�sentation sch�matique de SINDy.
        <with|font-shape|<quote|italic>|Brunton et al. 2015>
      </surround>|<pageref|auto-7>>

      <tuple|normal|<\surround|<hidden-binding|<tuple>|2>|>
        Repr�sentation de SINDy-PI. <with|font-shape|<quote|italic>|Kaheman
        et al. 2020>
      </surround>|<pageref|auto-13>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Mod�lisation
      fondamentale> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|3tab>|Probl�me de minimisation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|3tab>|Sequential Thresholded Least Squares
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|3tab>|Dimension
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|3tab>|EDO � param�tres, forcing, contr�le
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|3tab>|Une illustration
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>PDE-FIND>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>SINDy
      with sparse regularized regression>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc><locus|<id|%1233163F8-119D43A00>|<link|hyperlink|<id|%1233163F8-119D43A00>|<url|https://arxiv.org/abs/2004.02322>>|SINDy-PI>,
      une m�thode encore plus efficace> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10><vspace|0.5fn>

      <with|par-left|<quote|3tab>|Implicit SINDy
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11>>

      <with|par-left|<quote|3tab>|SINDy-PI
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-12>>
    </associate>
  </collection>
</auxiliary>