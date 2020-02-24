from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
if __name__ == "__main__":
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')

    text = """
    À défaut de neige, c'est une avalanche de critiques qui a déferlé sur la station de Luchon-Superbagnères (Haute-Garonne) depuis le 14 février. Après avoir décidé d'amener de la neige par hélicoptère en bas de ses pistes qui en manquaient cruellement, la petite station des Pyrénées s'est retrouvée malgré elle au centre d'une polémique remontée jusqu'à la ministre de l'Écologie. « Enneiger les stations de ski par hélicoptère n'est pas une voie possible », a tonné Élisabeth Borne sur Twitter.  Un peu sonnée, la station assume. « Cette action a été réfléchie », assure à France Info son directeur, Christian Mathias, pointant un « hiver particulier et difficile ». Même son de cloche du côté du département, qui gère la station en régie : « Nous avons voulu sauver de l'emploi, je le revendique », lâche Georges Méric, président du conseil départemental de Haute-Garonne, à France Bleu. « Et si vous parlez du bilan carbone de la station, cette année il sera meilleur qu'en 2019 puisque nos dameuses ne travaillent pas. »   closevolume_off  En faisant déplacer de la neige par hélicopère les 14 et 15 février dernier, Luchon-Superbagnères, dans les Pyrénées, a créé une polémique remontée jusqu'au gouvernement.  © ANNE-CHRISTINE POUJOULAT / AFP Raréfaction de la neige Aux côtés de Luchon, les stations concurrentes font bloc et dénoncent une polémique « inutile ». « C'est tellement facile de nous montrer du doigt », lâche Jean-Pierre Rougeaux, maire de Valloire (Savoie), joint par Le Point. « Nous sommes un peu le thermomètre du changement climatique, mais on n'en est pas responsables. C'est toute la planète qui en est responsable ! »   Et si la foudre ministérielle est tombée sur la petite station des Pyrénées, elle aurait pu tout autant en toucher d'autres. Car la neige se fait de plus en plus rare, et cet hiver est particulièrement compliqué. Dans les Alpes, Montclar les Deux Vallées a aussi eu recours à des livraisons de neige par hélicoptère fin décembre pour sauver sa deuxième semaine des vacances de Noël. Une semaine avant Luchon, c'est Gérardmer, dans les Vosges, qui avait fait livrer de l'or blanc par camion.  Et la situation ne risque pas de s'arranger. « Les régions de montagne sont plus touchées que les plaines par le changement climatique : l'élévation de température y est en général plus forte qu'en moyenne sur la planète », prévient Météo-France. L'effet du réchauffement climatique devrait surtout se faire sentir à moyenne altitude, entre 1 200 et 2 000 mètres.  Lire aussi Réchauffement : la décennie écoulée a été la plus chaude jamais enregistrée   Les relevés de hauteur de neige effectués par Météo-France depuis 1960 à 1 325 mètres d'altitude au col de Porte, dans les Alpes – considérés comme une référence pour l'étude de l'enneigement à moyenne altitude en France – le montrent bien : si les hivers sans neige ont toujours existé, ils sont de plus en plus fréquents. Les hivers avec une grande quantité de neige sont, eux, de plus en plus rares. Et quand la neige est présente, c'est en moindre quantité, et pendant moins longtemps.   « On ne peut pas jouer contre la nature » À la ministre, qui a estimé dans 20 Minutes qu'« on ne peut pas jouer contre la nature », les stations, elles, opposent la pression économique à laquelle elles sont confrontées. À Montclar, le gestionnaire explique au Dauphiné libéré que l'intervention de l'hélicoptère a représenté un investissement de 8 000 euros, pour assurer l'ouverture sur une semaine de vacances qui devait rapporter 150 000 euros au seul exploitant, sans compter les retombées sur les commerçants. Le calcul a été vite fait.   En France, dans le top 3 mondial de fréquentation après les États-Unis et l'Autriche, le ski représente 18 000 emplois directs – permanents ou saisonniers –, 120 000 en tout, et 10 milliards d'euros de dépenses en station chaque hiver, selon Domaines skiables de France. Difficile de se défaire de la pratique des sports d'hiver quand elle est l'une des plus importantes sources de revenus en montagne. Face au changement climatique, les stations de ski ne semblent donc pas avoir encore pris la mesure du défi qui les attend.  Lire aussi France : la fréquentation touristique rebondit  Le changement climatique a pourtant déjà fait des victimes. À Saint-Honoré 1500, en Isère, les immeubles abandonnés en pleine construction témoignent de la chute brutale de cette petite station ouverte dans les années 1980 par une liaison avec le domaine skiable voisin de l'Alpe du Grand Serre. Déjà plombée par des problèmes financiers, la station, trop basse et exposée plein sud, manquait cruellement de neige. Les remontées mécaniques ont été fermées en 2003.   Depuis la fermeture des remontées mécaniques en 2003, la station Saint-Honoré 1500, dans les Alpes, est figée dans le temps.  © MAXPPP / PHOTOPQR/LE PROGRES Les stations à plus faible altitude sont les plus touchées par le changement climatique, et les plus faibles comme Saint-Honoré, qui n'ont pas les moyens d'investir pour se transformer, en sont les premières victimes. Depuis 1951, ce sont 169 stations qui ont mis la clé sous la porte, près de la moitié à cause du manque de neige, a calculé Pierre-Alexandre Métral, doctorant en géographie alpine à l'université de Grenoble. « Pendant trente ans, c'était des domaines skiables équipés de deux ou trois remontées mécaniques qui fermaient, explique-t-il à Slate. Mais depuis les années 2000, des stations de plus en plus grandes qui comptent parfois entre sept et dix remontées mécaniques ont mis la clé sous la porte. »  Course aux flocons Élisabeth Borne l'a reconnu : « Il faut un plan d'action pour accompagner les stations face au dérèglement climatique. » Mais existe-t-il seulement des solutions pour les sauver d'un phénomène global sur lequel elles n'ont aucune prise ? Bien souvent, c'est vers la neige de culture que les exploitants se sont tournés. Certes, cette neige a un coût (2,50 euros pour un mètre cube), consomme de l'électricité (1 à 3 kWh par mètre cube) et de l'eau (un mètre cube d'eau permet de produire le double de neige). Mais combinée à des dameuses intelligentes qui permettent de mieux répartir la neige, et à des pistes plus lisses qui nécessitent beaucoup moins d'enneigement pour être praticables, la neige de culture permet de limiter la casse et de garantir l'ouverture de la station.  La plupart des stations sont aujourd'hui équipées en canons à neige, et y consacrent 6 à 10 % du prix du forfait. Le prix à payer pour garantir l'activité. Dans une étude menée entre 2017 et 2018 sur 23 stations, le département de l'Isère estimait que 42 % de la surface de leurs domaines skiables seraient équipés pour la neige de culture d'ici à 2025, ce qui permettrait de « maintenir un niveau d'enneigement en 2050 similaire à celui d'aujourd'hui ».   Pour pallier au manque de neige, la plupart des stations se sont dotées de canons à neige, comme ici à Font-Romeu, dans les Pyrénées. Une solution de court-terme seulement.  © RAYMOND ROIG / AFP Mais cette solution miracle ne marchera pas sur le long terme. Pour faire de la neige, même artificielle, il faut qu'il fasse moins de zéro, ce qui, à l'horizon post-2050, n'est pas gagné. « Avec un taux de couverture par la neige de culture de 45 %, l'enneigement demeure comparable à la situation actuelle pour un réchauffement planétaire inférieur à deux degrés, mais au-delà de trois degrés, la neige de culture ne suffit plus à compenser la réduction de l'enneigement naturel », estime Météo-France, se basant sur une étude parue dans Scientific Reports.  Extension des domaines skiables Au lieu de s'acharner à fabriquer de la neige là où il fait trop chaud, certaines stations ont opté pour des extensions de leurs domaines vers le haut, à des altitudes qui souffrent moins du réchauffement et du manque de neige. Cet hiver, Valloire a inauguré un nouveau télésiège et deux pistes rouges dans un secteur encore non exploité, portant ainsi le point culminant du domaine skiable à 2 750 mètres. Pour préserver au maximum le relief de la montagne, et se préserver des critiques des militants écologiques, le terrassement des deux nouvelles pistes a été limité.  En Maurienne (Savoie), on a même ressorti des cartons un projet vieux de plus de 30 ans, pour l'inscrire à nouveau au schéma de cohérence territoriale (SCoT). « La Croix du Sud », une extension de près d'une centaine d'hectares pour les domaines skiables de Valmeinier et Valfréjus. Une nécessité pour survivre, explique à France Bleu Jean-Claude Raffin, vice-président du SCoT Pays de Maurienne : « Compte tenu du changement climatique, l'idée, c'est bien de pouvoir remonter les domaines skiables en altitude pour pérenniser encore nos stations pendant quelque temps. »  Lire aussi Cour des comptes : les stations de ski poussées à se réinventer  Une hérésie pour Annie Collombet, président de l'association Vivre et agir en Maurienne, qui regrette que l'« on continue dans les vieux schémas ». L'avis de la Mission régionale d'autorité environnement sur plusieurs projets de ce type en Auvergne-Rhône-Alpes n'est guère plus enthousiaste : « En l'état, certains de ces projets, tels que l'interconnexion de la Croix du Sud, l'extension des domaines skiables de Val Cenis ou d'Aussois, sont susceptibles de causer des dommages très significatifs, voire irréversibles, à des milieux écologiques d'une valeur exceptionnelle. »  Trouver un nouveau modèle Plutôt que de s'engager dans une bataille perdue d'avance contre le climat, d'autres ont préféré prendre les devants pour sécuriser leur activité sur le long terme en développant des modèles « quatre saisons ». L'objectif : casser le monopole du ski, qui attire encore la plus grande partie de la clientèle, en développant de nouvelles activités susceptibles de prendre sa succession le jour où la neige viendra à manquer. La ministre de l'Écologie a d'ailleurs poussé en ce sens jeudi 20 février, lors d'une réunion avec les représentants de stations après la polémique de Luchon. « Il faut accélérer cette transition », a-t-elle indiqué à l'Agence France Presse, promettant une « offre complète d'accompagnement des stations pour à la fois encourager leurs pratiques vertueuses en termes d'environnement et les aider à s'adapter ».  Un véritable défi car, pour le moment, la saison estivale ne représente que 5 % du chiffre d'affaires annuel des stations ouvertes à cette période. Plus de la moitié n'arrive même pas à couvrir leurs charges de fonctionnement pendant l'été. « Aujourd'hui, on n'a aucun produit pour remplacer la pente enneigée à dévaler, à ski, en snow ou autre », résume au Point le maire de Valloire. Pour opérer un véritable basculement, c'est tout le modèle économique des stations qu'il faut revoir.Plus les années passent, plus la neige monte, on le voit bien. À Chamrousse, la mairie a lancé le projet « Chamrousse 2030 », qui envisage de faire du village une « smart station d'altitude quatre saisons ». Au programme : une grande rénovation de Chamrousse 1650, avec construction de nouveaux bâtiments plus modernes que les barres que l'on trouve habituellement en station. L'objectif affiché est de faire venir plus d'habitants à l'année pour que les commerces puissent fonctionner sans dépendre du tourisme hivernal, développer le tourisme d'affaires et les séjours courts, et ouvrir une piste de luge d'été et un centre aquatique pour compléter l'offre hors neige (VTT, randonnées…) déjà existante. Encore faut-il avoir les moyens d'investir. Dans les Pyrénées-Orientales, la petite station municipale du Puigmal a fermé en 2013, ruinée par plusieurs saisons sans neige. « Plus les années passent, plus la neige remonte, on le voit bien », assure au Point Isidore Peyrato, premier adjoint d'Err, la commune sur laquelle était implantée la station. Elle a finalement rouvert fin 2019 sous une nouvelle forme : circuit de randonnée à ski, trail, pistes de VTT… C'est l'entreprise Rossignol, spécialiste du matériel et vêtements des sports d'hiver, qui a répondu à l'appel à projets de la mairie. « Ils ne gèrent pas la station, mais ils nous aident à mettre en place les parcours, notamment sur le plan technique, et assurent la promotion », moyennant une redevance de la commune, explique Isidore Peyrato. La station du Puigmal, dans les Pyrénées, ici en novembre 2017, a fermé en 2013 après plusieurs saisons sans neige. Elle a rouvert cet hiver sous une nouvelle forme. Les remontées mécaniques, elles, restent pour l'instant à l'arrêt. Les hivers étant devenus trop incertains, Err-Puigmal ne souhaite pas vraiment se relancer dans la gestion de l'infrastructure lourde d'une station de ski. Un loueur de matériel s'est à nouveau installé, et les élus espèrent bien voir rouvrir les anciens restaurants, aujourd'hui à l'abandon en bas des pistes. « La priorité pour l'instant, c'est de faire revivre l'économie. » À terme, les télésièges pourraient être redémarrés s'ils s'avèrent utiles pour remonter les VTT par exemple. Mais la mairie l'assure, à Puigmal, le ski alpin, c'est de l'histoire ancienne.
    """
    text = text + """ 15 février 2020. – Ce vendredi 14 février, à l'hôpital Hôpital Bichat-Claude-Bernard de Paris, un homme est décédé du nouveau coronavirus COVID-19. Il s'agit d'un touriste chinois de 80 ans, originaire de la province du Hubei, qui avait été hospitalisé en fin janvier avec des symptômes du virus. Le ministre de la santé Agnès Buzyn a annoncé la mort du Chinois au samedi. La fille du Chinois décédé a aussi été hopitalisée, mais elle semble guérie maintenant. C'est la première fois en Europe qu'une personne meurt du nouveau virus, qui s'est répandu à partir de Chine depuis décembre 2019. C'est aussi le premier cas mortel suite au COVID-19 qui se produit hors d'Asie. Au samedi, un douzième cas du virus a été identifié en France."""
    text = text + """ La crise américano-iranienne de 2019-2020 est un conflit asymétrique opposant les États-Unis et l'Iran du 27 décembre 2019 au 9 janvier 2020. Principaux alliés du gouvernement irakien lors du conflit mené contre l'État islamique entre 2013 et 2017, les États-Unis et l'Iran se déchirent rapidement après la proclamation de la victoire sur les djihadistes. En 2018, Washington se retire unilatéralement de l'Accord de Vienne sur le nucléaire iranien et rétablit ses sanctions contre l'Iran. Téhéran accroît pour sa part son influence en Irak, notamment par le biais de milices chiites des Hachd al-Chaabi soutenues par les Gardiens de la révolution islamique. En juin 2019, l'Iran abat un drone américain au-dessus du détroit d'Ormuz. Fin 2019, les États-Unis accroissent les sanctions économiques, l'Iran et l'Irak sont touchés par de vastes manifestations anti-gouvernementales, des milices pro-iraniennes mènent des attaques en Irak contre des cibles américaines et l'armée américaine attaque officiellement des cibles militaires iraniennes. L'ensemble génère des réactions internationales importantes : des soutiens aux deux parties prenantes, des demandes d'apaisement et une plainte auprès de l'ONU. En 2018, le retrait unilatéral des Américains sur l'Accord de Vienne sur le nucléaire iranien et le rétablissement des sanctions contre l'Iran provoque également une dégradation des relations entre les États-Unis et l'Irak. En août 2019, le président français Emmanuel Macron tente de faire baisser les tensions lors du sommet du G7 de Biarritz en invitant le ministre des affaires étrangères iranien. Mais en septembre, après l'attaque d'Abqaïq et de Khurais, le président américain Donald Trump accentue les sanctions économiques en déclarant qu'elles sont « les plus sévères jamais imposées à un pays »."""
    text = text + """ Ce dimanche, dans les rues de Bruxelles, à peu près 8.000 manifestants ont pris part à une nouvelle "marche pour le climat". Ils ont protesté contre le rejet, il y a deux jours, par le gouvernement belge de la proposition de révision de l'Article 7 bis de la Constitution belge. Cette révision de la loi visait à atteindre une politique climatique plus efficace. Vers 14h00, les manifestants sont partis de la Gare du Nord. Vers 16h00, ils sont arrivés au parc du Cinquantenaire. Une heure plus tôt, à la hauteur de la rue de la Loi, une délégation des "gilets jaunes", venus en particulier de la France et des Pays-Bas, avait quitté le cortège pour faire des dégâts, brisant les vitres d’un bâtiment. La police de Bruxelles Capitale Ixelles déclare avoir interpellé à peu près 70 personnes. En plus, il y avait à peu près 7000 manifestants à Liège aujourd'hui."""
    text = text + """ Le Hirak (en arabe : الحراك, Mouvement) désigne une série de manifestations sporadiques qui ont lieu depuis le 16 février 2019 en Algérie pour protester dans un premier temps contre la candidature d'Abdelaziz Bouteflika à un cinquième mandat présidentiel, puis contre son projet, également contesté par l'armée, de se maintenir au pouvoir à l'issue de son quatrième mandat dans le cadre d'une transition et de la mise en œuvre de réformes. Par la suite, les protestataires réclament la mise en place d'une Deuxième République, et le départ des dignitaires du régime, notamment parce que ceux-ci organisent le prochain scrutin avec les candidatures de caciques du régime, ce qui mène à l'élection de l'ancien Premier ministre Abdelmadjid Tebboune, lui-même contesté par les manifestants. D'une ampleur inédite depuis des décennies, ces manifestations, qui ont essentiellement lieu les vendredis et mardis (pour les étudiants), conduisent Bouteflika à démissionner le 2 avril 2019, après la défection de l'Armée nationale populaire, qui s'opposait au projet de Bouteflika de se maintenir au pouvoir au-delà de son mandat dans le cadre d'une transition et de réformes. Celui-ci est remplacé par intérim par Abdelkader Bensalah. Les manifestants continuent cependant à se mobiliser afin d'obtenir la mise en place d'une transition et la nomination d'un président et d'un gouvernement de consensus, ce que rejette l'armée, arguant que cette proposition serait inconstitutionnelle et source d'instabilité. L'armée rejette également toute transition, que ce soit en convoquant une assemblée constituante, ou des législatives anticipées, ou tout départ de l'équipe exécutive sortante. """
    text = text + """ La crise présidentielle au Venezuela est une crise politique autour de la légitimité de la présidence du pays depuis le 10 janvier 2019. Il existe une opposition entre Nicolás Maduro, Président en fonction depuis la séparation du pouvoir et disparition physique de Hugo Chavez, et Juan Guaidó Président du Parlement3 depuis le 5 janvier 2019. Juan Guaidó s'est proclamé Président Encargado du pays devant une manifestation de dizaines de milliers de personnes dans un cabildo abierto. Il a décidé de ce faire profitant qu'il était Président du parlement et dû au manque de reconnaissance ou de légitimité de la dernière élection présidentielle un an avant. Il évoqua la Constitution dans ses articles 233, 333 et 350. Il est reconnu par la plupart de l'opposition au régime de Maduro comme le Président (E) du Venezuela et bénéficie de la reconnaissance d'une soixantaine de nations (dont l'Allemagne, l'Australie, les États-Unis, la France, le Japon et le Royaume-Uni). Cependant, Nicolás Maduro est reconnu comme étant le Chef de l'État ou le Président par les Pro-Chavez (ou "chavistas") et bénéficie de la reconnaissance d'une vingtaine de pays (dont la Chine, l'Iran, le Mexique et la Russie). Le restant des pays du monde ne se prononcent toujours pas sur cette crise présidentielle ni pour l'un ou pour l'autre, ou ont plutôt une position de neutralité."""
    text = text + """ scolaire, le vendredi ou le jeudi, pour participer à des manifestations en faveur de l’action contre le réchauffement climatique. La première grève scolaire pour le climat a été lancée par Greta Thunberg, le 20 août 2018 devant le Riksdag (Parlement suédois). L'adolescente militante suédoise explique aux journalistes conviés qu'elle n'ira pas à l'école jusqu'aux élections générales du 9 septembre 2018. Elle continuera, après les élections, à faire grève chaque vendredi, attirant ainsi l'attention du monde entier sur le mouvement baptisé Fridays for Future. Suite à cela, la mobilisation prend une très grande ampleur en Belgique, réunissant des dizaines de milliers de manifestants chaque jeudi depuis janvier 2019 dans les rues de Bruxelles et des autres villes du pays, à tel point que les absences des élèves et des étudiants sont tolérées dans plusieurs écoles et universités du pays. Ces grèves — dénommées « école buissonnière pour le climat » par certains médias — sont organisées par différents mouvements tels que « Grève de la Jeunesse pour le Climat et l'Environnement (Youth for Climate) », Student for Climate, etc."""
    sentences = text.split(".")

    corpus_embedding = embedder.encode(sentences)

    # Perform kmean clustering
    inertia_list = []
    K = range(1, 20)
    for k in K:
        print("{} clusters ---------------------".format(k))
        num_clusters = k
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embedding)
        cluster_assignment = clustering_model.labels_
        inertia_list.append(clustering_model.inertia_)

        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(sentences[sentence_id])

        for i, cluster in enumerate(clustered_sentences):
            print("Cluster ", i + 1)
            print(cluster)
            print("")

    plt.plot(K, inertia_list, "bx-")
    plt.xlabel("k")
    plt.ylabel("Sum of squared distances")
    plt.title("Elbow method for optimal k")
    plt.show()
    plt.savefig("./elbow_cluster.png")