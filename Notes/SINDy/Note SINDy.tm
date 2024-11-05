<TeXmacs|2.1.4>

<style|generic>

<\body>
  <doc-data|<doc-title|Note SINDy>>

  SINDy, ou \PSparse Identification of Nonlinear Dynamics\Q, est une méthode
  d'identification automatique de modèle à partir de données, reposant sur un
  problème de minimisation construit pour assurer autant la fidélité aux
  données du modèle obtenu ainsi que sa \Psparsité\Q (modèle parcimonieux).

  Le <hlink|modèle original|https://arxiv.org/abs/1509.03580> de Brunton et
  al. (2015) a connu plusieurs variantes :
  <hlink|PDE-FIND|https://arxiv.org/abs/1609.06401>, <hlink|SINDy with
  control|https://www.sciencedirect.com/science/article/pii/S2405896316318298?ref=cra_js_challenge&fr=RR-1>,
  <hlink|SINDy with sparse relaxed regularized
  regression|https://arxiv.org/abs/1906.10612> et
  <with|font-series|bold|<hlink|SINDy-PI|https://arxiv.org/abs/2004.02322>>
  (implicit-SINDy), que nous détaillons plus loin.

  <section|Modélisation fondamentale>

  On suppose que l'on dispose de données temporelles pour une fonction
  <math|x<around*|(|t|)>> sous la forme de deux vecteurs,
  <math|<with|font-series|bold|x>> et <math|<wide|<with|font-series|bold|x>|\<dot\>>>
  de taille <math|N>. On suppose aussi qu'il existe un <math|f> tel que

  <\equation>
    <wide|x|\<dot\>><around*|(|t|)>=f<around*|(|x<around*|(|t|)>|)>,
  </equation>

  que l'on va tenter de reconstruire à partir des données. On considère une
  bibliothèque composée de <math|K> fonctions candidates

  <\equation*>
    \<Theta\>=<matrix|<tformat|<table|<row|<cell|f<rsub|1> f<rsub|2>
    \<ldots\>>>>>>,
  </equation*>

  et l'on supposera que <math|f> s'écrit comme
  <with|font-shape|italic|combinason linéaire parcimonieuse> de fonctions de
  cette bibliothèque. On construit la matrice de taille <math|N\<times\>K>
  associée à la bibliothèque et aux données :

  <\equation*>
    <with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>=<matrix|<tformat|<table|<row|<cell|\<theta\><rsub|1><around*|(|<with|font-series|bold|x<rsub|1>>|)><space|1em>\<theta\><rsub|2><around*|(|<with|font-series|bold|x<rsub|1>>|)>
    \<ldots\>>>|<row|<cell|\<theta\><rsub|1><around*|(|<with|font-series|bold|x<rsub|2>>|)><space|1em>\<theta\><rsub|2><around*|(|<with|font-series|bold|x<rsub|2>>|)>
    \<ldots\>>>|<row|<cell|\<vdots\><space|3em>\<vdots\><space|2em>>>>>>,
  </equation*>

  et le problème revient maintenant à trouver un vecteur
  <math|\<xi\>\<in\>\<bbb-R\><rsup|K>> de norme
  <math|\<\|\|\>\<cdot\>\<\|\|\><rsub|0>> la plus petite possible et qui
  minimise \<\|\|\><math|<wide|<with|font-series|bold|x>|\<dot\>>> <math|->
  <math|<with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>\<xi\>>\<\|\|\>.\ 

  <paragraph|Problème de minimisation>

  Précisément, le problème initial s'écrit donc

  <\equation*>
    <below|argmin|\<xi\>\<in\> \<bbb-R\><rsup|K>>
    \<\|\|\><wide|<with|font-series|bold|x>|\<dot\>>-<with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>\<xi\>\<\|\|\><rsub|2><space|0.6spc>+<space|0.6spc>\<lambda\>\<\|\|\>\<xi\>\<\|\|\><rsub|0>,
  </equation*>

  où <math|\<lambda\>\<geq\> 0> es un paramètre de régularisation. Ce
  problème est néanmoins np-dur, ainsi on pourra le relaxer en le LASSO

  <\equation*>
    <below|argmin|\<xi\>\<in\> \<bbb-R\><rsup|K>>
    \<\|\|\><wide|<with|font-series|bold|x>|\<dot\>>-<with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>\<xi\>\<\|\|\><rsub|2><space|0.6spc>+<space|0.6spc>\<lambda\>\<\|\|\>\<xi\>\<\|\|\><rsub|1>.
  </equation*>

  <paragraph|Sequential Thresholded Least Squares>

  Le LASSO peut devenir cher computationellement pour les larges datasets, et
  de plus n'est pas très adapté à la sélection de coefficients. Une
  alternative proposée, qui se trouve de plus être simple et robuste au
  bruit, est la méthode des <with|font-shape|italic|Sequential Thresholded
  Least Squares>, où l'on imposera la sparsité \Pà la main\Q, en effectuant
  récursivement une régression des moindres carrés, c'est à dire en résolvant
  <math|<below|argmin|\<xi\>\<in\> \<bbb-R\><rsup|K>>
  \<\|\|\><wide|<with|font-series|bold|x>|\<dot\>>-<with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>\<xi\>\<\|\|\><rsub|2>>,
  puis en éliminant de la libraire <math|\<Theta\>> les fonctions pour
  lesquelles le coefficient associé est plus faible qu'un certain cut-off
  <math|\<lambda\>>. L'identification de ce paramètre peut s'effectuer par de
  la <with|font-series||validation> croisée.

  <paragraph|Dimension>Si <math|x> n'est pas à valeurs dans <math|\<bbb-R\>>
  mais <math|\<bbb-R\><rsup|d>> avec <math|d> grand, la méthode SINDy
  deviendra moins adaptée et il faudra recourir à des méthodes de réduction
  de dimension. Pour notre problème, il faudra voir si c'est un souci ou non
  : ça dépendra de comment l'on gère les interactions.

  <paragraph|EDO à paramètres, forcing, contrôle>Il est possible d'étendre
  SINDy de <math|<around*|(|1|)>> à des équations à paramètres, dependantes
  du temps ou avec <hlink|forçage|https://www.sciencedirect.com/science/article/pii/S2405896316318298?ref=cra_js_challenge&fr=RR-1>
  simplement en ajoutant des fonctions dans la librairie.

  <paragraph|Une illustration>

  <\big-figure|<image|/Users/thomasborsoni/Desktop/Post-doc/Projet
  fourmis/Programmes/Notes/SINDy/SINDy_1.png|1par|||>>
    Représentation schématique de SINDy. <with|font-shape|italic|Brunton et
    al. 2015>
  </big-figure>

  \;

  <section|PDE-FIND>

  Une extension à SINDy pour retrouver des EDP a aussi été proposée dans
  <hlink|cet article|https://arxiv.org/abs/1609.06401>. L'extension est assez
  naturelle : supposons que l'on étudie un fonction
  <math|u<around*|(|t,x|)>>, satisfaisant une EDP de la forme

  <\equation*>
    \<partial\><rsub|t>u=N<around*|(|u,\<partial\><rsub|x>u,\<partial\><rsup|2><rsub|x
    x>u,\<ldots\>,x,t|)>,
  </equation*>

  et que l'on cherche à retrouver la fonction <math|N>. La méthode PDE-FIND
  est une variante de SINDy où la bibliothèque <math|\<Theta\>> contiendra
  des fonctions de <math|u>, <math|\<partial\><rsub|x>u>,
  <math|\<partial\><rsup|2><rsub|x x>u>, <math|x> et <math|t>.

  <section|SINDy with sparse regularized regression>

  La méthode SR3 proposée dans <hlink|cet
  article|https://arxiv.org/abs/1906.10612> propose une alternative à la
  méthode de Sequential Thresholded Least Squares (STLSQ). Elle en est une
  sorte de généralisation, bien que STLSQ ne soit pas réellement incluse.

  Le problème de minimisation étudié sera ici

  <\equation*>
    <below|argmin|W\<in\> \<bbb-R\><rsup|K>><space|1em><below|min|\<xi\>\<in\>
    \<bbb-R\><rsup|K>> <frac|1|2>\<\|\|\><wide|<with|font-series|bold|x>|\<dot\>>-<with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>|)>\<xi\>\<\|\|\><rsup|2><rsub|2><space|0.6spc>
    +<space|0.6spc> <frac|1|2 \<nu\>>\<\|\|\>\<xi\>-W\<\|\|\><rsup|2><rsub|2><space|0.6spc>
    +<space|0.6spc> \<lambda\> R<around*|(|W|)>
  </equation*>

  Si l'on prend comme fonction <math|R> la norme <math|\<ell\><rsup|0>>, ce
  problème sera similaire à STLSQ. On peut commencer avec STLSQ, puis passer
  à SR3 si l'on est limité.

  <section|<hlink|SINDy-PI|https://arxiv.org/abs/2004.02322>, une méthode
  encore plus efficace>

  Le potentiel problème central de la méthode SINDy est que l'on doit
  supposer que la fonction <math|f> est une
  <with|font-shape|italic|combinaison linéaire> d'éléments de la bibliothèque
  <math|\<Theta\>>, ce qui restreint fortement les cas pour lesquels la
  méthode pourrait fonctionner.

  L'idée proposée est alors de plutôt étudier le problème dit \Pimplicite\Q

  <\equation*>
    f<around*|(|x,<wide|x|\<dot\>>|)>=0,
  </equation*>

  pour lequel on va alors tenter de reconstruire <math|f> comme
  <with|font-shape|italic|combinaison linéaire parcimonieuse> de fonctions
  appartenant à la librairie <math|\<Theta\><around*|(|x,<wide|x|\<dot\>>|)>>.
  À noter que le cas standard se retrouve avec la fonction
  <math|f<around*|(|x,y|)>=y-g<around*|(|x|)>>.

  Dans ce cas, on considère alors une librairie <math|\<Theta\>> contenant
  des fonctions de <math|x> et de <math|<wide|x|\<dot\>>>, par exemple
  <math|<around*|{|1,x,x<rsup|2>,<wide|x|\<dot\>>,x<wide|x|\<dot\>>,x<rsup|2><wide|x|\<dot\>>|}>>.
  Notons que cette librairie a alors accès aux équations comme
  <math|<wide|x|\<dot\>>=<frac|x|1+x>>.

  <paragraph|Implicit SINDy>Le problème, dans le cadre de SINDy, est alors de
  trouver le vecteur <math|\<xi\>> le plus sparse possible et tel que l'on
  ait

  <\equation*>
    <with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>,<with|font-series|bold|<wide|x|\<dot\>>>|)>
    \<xi\>\<approx\>0.
  </equation*>

  Ce problème est néanmoins mal posé, l'équation ci-dessus étant invariante
  par la multiplication d'un constante. Il faut donc fixer la valeur d'une
  norme de <math|\<xi\> ><math|<around*|(|par exemple sa norme 2|)>>. On
  obtient alors le problème de minimisation

  <\equation*>
    <below|argmin|\<xi\>><space|1em>\<\|\|\><with|font-series|bold|\<Theta\>><around*|(|<with|font-series|bold|x>,<with|font-series|bold|<wide|x|\<dot\>>>|)>
    \<xi\>\<\|\|\><rsub|2>+\<lambda\>\<\|\|\>\<xi\>\<\|\|\><rsub|p>,<space|1em>
    <space|1em> \<\|\|\>\<xi\>\<\|\|\><rsub|2>=1,
  </equation*>

  où en théorie on souhaiterait <math|p=0> mais l'on mettrait en pratique
  <math|p=1>. La résolution de ce problème se trouve être difficile, à cause
  de la contrainte, et l'on peut en fait appliquer une méthode plus efficace
  et robuste au bruit.

  <paragraph|SINDy-PI>L'accronyme \PPI\Q vaut pour \PParallel, Implicit\Q.

  L'idée de SINDy-PI est de <with|font-shape|italic|traiter le problème
  implicite> décrit plus haut <with|font-shape|italic|avec la méthode
  explicite> traditionnelle. Considérons une librairie
  <with|font-series|bold|<math|\<Theta\><around*|(|x,<wide|x|\<dot\>>|)>>>.
  Pour toute colonne <math|<with|font-series|bold|\<theta\>><rsub|j>> de
  <math|<with|font-series|bold|\<Theta\>>>, on va résoudre le problème SINDy
  standard associé à l'équation

  <\equation*>
    <with|font-series|bold|\<theta\>><rsub|j>=<with|font-series|bold|\<Theta\>><rsup|j>
    <with|font-series||\<xi\>><rsup|j>,
  </equation*>

  où <math|<with|font-series|bold|\<Theta\>><rsup|j>> est la librairie
  <with|font-series|bold|<math|\<Theta\>>> à laquelle on a retiré la colonne
  <math|j>. Par suite, on fait la chose suivante : si le vecteur
  <math|\<xi\><rsup|j>> sélectionné n'est pas sparse et/ou donne une mauvaise
  prédiction, on rejette la fonction <math|<with|font-series|bold|\<theta\>><rsub|j>>
  et l'on recommence après avoir retiré <math|<with|font-series|bold|\<theta\>><rsub|j>>
  de la librairie. D'un autre côté, si l'on obtient un vecteur sparse et qui
  donne une bonne prédiction, on peut s'arrêter.

  On peut aussi considérer <with|font-series||de<with|font-series||ux>>
  librairies distinctes pour les termes de gauche et de droite.

  <\big-figure|<image|/Users/thomasborsoni/Desktop/Post-doc/Projet
  fourmis/Programmes/Notes/SINDy/SINDy_2.png|1par|||>>
    Représentation de SINDy-PI. <with|font-shape|italic|Kaheman et al. 2020>
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
        Représentation schématique de SINDy.
        <with|font-shape|<quote|italic>|Brunton et al. 2015>
      </surround>|<pageref|auto-7>>

      <tuple|normal|<\surround|<hidden-binding|<tuple>|2>|>
        Représentation de SINDy-PI. <with|font-shape|<quote|italic>|Kaheman
        et al. 2020>
      </surround>|<pageref|auto-13>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Modélisation
      fondamentale> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|3tab>|Problème de minimisation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|3tab>|Sequential Thresholded Least Squares
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|3tab>|Dimension
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|3tab>|EDO à paramètres, forcing, contrôle
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
      une méthode encore plus efficace> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
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