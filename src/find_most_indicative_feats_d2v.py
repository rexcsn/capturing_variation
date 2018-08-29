import argparse
import json
import multiprocessing
import sys
from collections import defaultdict
from itertools import islice, count
import spacy
from gensim.models import Doc2Vec
from gensim.models.doc2vec import FAST_VERSION
from gensim.models.doc2vec import TaggedDocument
from nltk import BigramCollocationFinder, BigramAssocMeasures, SnowballStemmer
from nltk.corpus import stopwords

assert FAST_VERSION > -1, "this will be painfully slow otherwise"

parser = argparse.ArgumentParser(description="find most indicative words using Doc2Vec")
parser.add_argument('--input', help='jodel archive file')
parser.add_argument('--min_tf', help='minimum term frequency for a feature to be counted', default=10, type=int)
parser.add_argument('--window', help='window size', default=15, type=int)
parser.add_argument('--save_corpus', help='save corpus after construction', default=None, type=str)
parser.add_argument('--load_corpus', help='load corpus', default=None, type=str)
parser.add_argument('--save_model', help='save trained model', default=None, type=str)
parser.add_argument('--limit', help='limit number of lines, for test purposes mainly', default=None, type=int)
args = parser.parse_args()

STOP_CITIES = {"höchstädt", "lahr/schwarzwald", "germersheim", "freiburg", "tauberbischofsheim", "rheinbach", "neuhaus",
               "drebkau", "steinheim", "florstadt", "marburg", "wörrstadt", "erlangen", "sitten", "eisenhüttenstadt",
               "flein", "horstmar", "braunlage", "homburg", "gerolstein", "goldkronach", "wolfratshausen", "meckenheim",
               "seftigen", "brandis", "thalheim", "schleusingen", "hatten", "naila", "hanau", "gröditz", "neuenrade",
               "neukalen", "netzschkau", "werlte", "breuberg", "gudensberg", "pfinztal", "bingen", "frechen",
               "schieder-schwalenberg", "augustusburg", "vlotho", "ebersbach", "arneburg", "kerpen", "mühltal", "neuss",
               "walsrode", "bornheim", "ohrdruf", "spenge", "meilen", "kyllburg", "lampertheim", "richtenberg",
               "prenzlau", "hemmoor", "daun", "trier", "greven", "bamberg", "berlin", "mahlberg", "gemeinde",
               "pirmasens", "dresden", "pattensen", "vaihingen", "lauffen", "feuchtwangen", "bern", "premnitz",
               "dassel", "haslach", "leutershausen", "werdau", "schüttorf", "vilsbiburg", "schwaan", "tönning",
               "murrhardt", "langenburg", "strasburg", "zeil", "schwerin", "nürtingen", "lotte", "lauter-bernsbach",
               "roggentin", "horb", "elsterwerda", "billerbeck", "oftersheim", "straelen", "budenheim", "sundern",
               "radeberg", "pulsnitz", "aichtal", "oberharz", "marl", "hennigsdorf", "rastatt", "oberlungwitz",
               "kevelaer", "puchheim", "walldorf", "frankfurt", "gößnitz", "luckenwalde", "leipzig", "hemau", "jarmen",
               "wriezen", "elsterberg", "zwönitz", "pegau", "wolkenstein", "schwelm", "prichsenstadt", "weener",
               "wanfried", "mengen", "wismar", "langenselbold", "gundelfingen", "blaubeuren", "sulz", "mühlacker",
               "bedburg", "donaueschingen", "lichtenstein", "dinslaken", "burglengenfeld", "biesenthal", "ludwigsburg",
               "niedernhall", "lenzburg", "aue", "goch", "immenhausen", "sachsenhagen", "waiblingen", "wittenburg",
               "kreuztal", "höhr-grenzhausen", "rosenheim", "oberpuchenau", "damme", "rahden", "torgelow",
               "lauterecken", "bleicherode", "waischenfeld", "wegberg", "steinhagen", "oldenburg", "schweinfurt",
               "dassow", "langenzenn", "leutenberg", "pottenstein", "horgen", "siegen", "schlieben", "hettstedt",
               "stendal", "sayda", "büttelborn", "naunhof", "goslar", "polch", "geiselhöring", "sinzig", "sinsheim",
               "emmerich", "brüssow", "bautzen", "mössingen", "eppelheim", "wildberg", "leimen", "haan",
               "wolframs-eschenbach", "geislingen", "kulmbach", "pfreimd", "lunzenau", "kamen", "rhens", "rathenow",
               "vechta", "welzheim", "ellingen", "greußen", "neumarkt", "suhl", "wien", "brüel", "hammelburg",
               "konstanz", "buchholz", "beeskow", "sarstedt", "gernsheim", "marienberg", "großbottwar", "saarburg",
               "töging", "biedenkopf", "gnoien", "geisa", "witzenhausen", "annweiler", "schmallenberg", "meyenburg",
               "nortorf", "hilchenbach", "uffenheim", "frauenfeld", "oettingen", "lauingen", "niemegk", "perleberg",
               "buchen", "dülmen", "lucka", "hillesheim", "hamm", "erbach", "bottrop", "wittingen", "weingarten",
               "tornesch", "burgbernheim", "viechtach", "kusel", "uetersen", "tübingen", "braunfels", "zwiesel",
               "boppard", "müllheim", "schenefeld", "winsen", "templin", "oebisfelde-weferlingen", "bacharach",
               "paderborn", "vetschau/spreewald", "horn-bad", "neumarkt-sankt", "hoya", "hollfeld", "lohne", "krems",
               "ludwigslust", "reichelsheim", "köthen", "köln", "ebermannstadt", "plattling", "haselünne", "schleiz",
               "gommern", "schotten", "hadamar", "lengede", "bünde", "stuhr", "meschede", "dornhan", "osterode",
               "kyritz", "butzbach", "mörfelden-walldorf", "kaub", "donauwörth", "westerstede", "eschwege", "stadtlohn",
               "aichach", "albstadt", "hohenstein-ernstthal", "düren", "barsinghausen", "bürstadt", "hermsdorf",
               "teupitz", "zörbig", "übach-palenberg", "magdala", "marktoberdorf", "kleve", "abensberg", "velburg",
               "backnang", "delmenhorst", "pforzheim", "fellbach", "zell", "bayreuth", "gützkow", "weißenstadt",
               "luckau", "möhrendorf", "großenhain", "markdorf", "riedenburg", "königs", "sondershausen", "warburg",
               "pegnitz", "ahlen", "schwerte", "döbeln", "ettenheim", "alsfeld", "morges", "rauschenberg", "ebern",
               "brotterode-trusetal", "moers", "castrop-rauxel", "selbitz", "nagold", "todtnau", "scheibenberg",
               "blieskastel", "germering", "verl", "neuenstadt", "ennigerloh", "ehingen", "erkner", "ortenberg",
               "korbach", "rastede", "döbern", "güstrow", "minden", "schauenstein", "aschaffenburg", "neuburg",
               "lörrach", "creuzburg", "ostheim", "teuschnitz", "neumark", "limbach-oberfrohna", "bockenem", "wehr",
               "amorbach", "tengen", "friesoythe", "diekholzen", "leer", "kirtorf", "eberswalde", "treuchtlingen",
               "fürstenau", "rottendorf", "heitersheim", "senden", "alsleben", "mülheim-kärlich", "nebra",
               "bernkastel-kues", "elsdorf", "schortens", "lauf", "lohfelden", "ueckermünde", "mittweida", "werne",
               "gelsenkirchen", "luzern", "trossingen", "kupferberg", "barby", "neustadt-glewe", "bischofswerda",
               "gernsbach", "reichenbach", "jena", "felsberg", "hermeskeil", "essen", "blaustein", "hitzacker",
               "mühldorf", "hagen", "bergkamen", "krempe", "rees", "rhinow", "neukirchen-vluyn", "meiningen",
               "laupheim", "schönberg", "sulingen", "bebra", "lehesten", "annaburg", "eisleben", "marsberg",
               "nördlingen", "rotenburg", "melsungen", "südliches", "stößen", "wolmirstedt", "parsberg", "garching",
               "schaffhausen", "dömitz", "graz", "barth", "nidda", "röbel/müritz", "salzwedel", "allendorf", "hagenow",
               "wilkau-haßlau", "werdenberg", "münchenbernsdorf", "spangenberg", "munster", "wertingen", "wernigerode",
               "genf", "sachsenheim", "gerolzhofen", "hamminkeln", "osterhofen", "annaberg-buchholz", "passau",
               "stolberg", "zülpich", "schwyz", "bendorf", "wil", "hauzenberg", "remseck", "thum", "ellrich",
               "hildburghausen", "weilburg", "bitterfeld-wolfen", "erding", "schkölen", "triberg", "diez", "delsberg",
               "brakel", "lauenburg/elbe", "fehmarn", "hemer", "gevelsberg", "vechelde", "kenzingen", "büren",
               "obernkirchen", "hemsbach", "hannover", "baruth/mark", "oberhausen", "neckarsulm", "königswinter",
               "weinstadt", "königsbrück", "trebbin", "großröhrsdorf", "diedorf", "fuldabrück", "hochheim", "greding",
               "aken", "betzdorf", "thaur", "apolda", "stadtlengsfeld", "groitzsch", "waldkraiburg", "dissen",
               "helmstedt", "endingen", "laufenburg", "schlettau", "mainburg", "ingolstadt", "harsum",
               "oestrich-winkel", "duderstadt", "alsdorf", "wetter", "hofheim", "herzberg", "diemelstadt", "traunreut",
               "strausberg", "heidenau", "geesthacht", "solms", "möckern", "potsdam", "kleinmachnow", "pfungstadt",
               "herrieden", "geretsried", "battenberg", "gaimersheim", "dornbirn", "ellwangen", "stadtoldendorf",
               "löningen", "seßlach", "dußlingen", "marlow", "wardenburg", "heusenstamm", "gera", "hausach",
               "neuötting", "langenau", "wittenberg", "calau", "nordenham", "hohenmölsen", "asperg", "lugau",
               "riedstadt", "anif", "fladungen", "reinheim", "darmstadt", "sonnewalde", "brake", "weißenthurm",
               "rodgau", "schleswig", "wolfenbüttel", "unna", "biberach", "meersburg", "rethem", "würzburg", "leuna",
               "rietberg", "montabaur", "plessur", "trebsen/mulde", "waldenbuch", "wülfrath", "rutesheim",
               "zerbst/anhalt", "hattingen", "heimsheim", "mendig", "trostberg", "demmin", "eutin", "püttlingen",
               "münstermaifeld", "steyr", "plau", "buckenhof", "alpirsbach", "zeitlarn", "dargun", "jessen",
               "freudenstadt", "plochingen", "langewiesen", "melle", "ramstein-miesenbach", "bremerhaven", "marktbreit",
               "amöneburg", "bühl", "pirna", "philippsburg", "verden", "alzenau", "lüneburg", "mücheln", "rodenberg",
               "fürstenwalde/spree", "jever", "immenstadt", "lemgo", "dreieich", "lengerich", "tann", "hofgeismar",
               "winterthur", "baden-baden", "cochem", "wedel", "leichlingen", "ebersberg", "rheinfelden", "münchberg",
               "kirchenlamitz", "wuppertal", "kraichtal", "longuich", "burgwedel", "gerabronn", "wildau", "rain",
               "dransfeld", "neunburg", "leingarten", "alzey", "erkrath", "schongau", "wertheim", "friedrichstadt",
               "schweich", "abenberg", "eltmann", "kitzingen", "werdohl", "auerbach/vogtl.", "wasungen", "adenau",
               "neustadt", "künzelsau", "frankenberg", "wilhelmshaven", "niefern-öschelbronn", "twistringen",
               "neustrelitz", "glückstadt", "tecklenburg", "hörstel", "süßen", "memmingen", "sulzburg", "stutensee",
               "schiltach", "bregenz", "rheinböllen", "chemnitz", "könnern", "kirchzarten", "harzgerode", "wesselburen",
               "babenhausen", "scheer", "reinbek", "grevesmühlen", "seesen", "warstein", "tegernsee", "hünfeld",
               "bobingen", "otterberg", "genthin", "osnabrück", "bürgel", "freudenberg", "hockenheim", "baunatal",
               "wiehe", "lippstadt", "dinkelsbühl", "stühlingen", "waren", "donzdorf", "wels", "werl", "greiz",
               "pritzwalk", "zella-mehlis", "heidenheim", "falkenstein/vogtl.", "marbach", "münster", "wittlich",
               "eisenach", "quickborn", "bernsdorf", "regis-breitingen", "vilshofen", "wermelskirchen", "scheinfeld",
               "schrobenhausen", "teltow", "steinach", "kandel", "ollmuth", "emden", "schöppenstedt", "lüdinghausen",
               "creußen", "kremmen", "schwalmstadt", "schönwald", "großbreitenbach", "berching", "windischeschenbach",
               "rheda-wiedenbrück", "clausthal-zellerfeld", "isny", "kranichfeld", "großräschen", "ostritz", "cottbus",
               "reinfeld", "mainz", "brunsbüttel", "güglingen", "grevenbroich", "altlandsberg", "gammertingen",
               "lütjenburg", "stans", "meppen", "geilenkirchen", "besigheim", "velten", "lindow", "gerbstedt",
               "lieberose", "malchow", "weinheim", "marktheidenfeld", "treuenbrietzen", "hechingen", "kamenz", "freren",
               "gräfenhainichen", "kirchheim", "wächtersbach", "bleckede", "vilseck", "plön", "syke", "löwenstein",
               "rheinau", "auma-weidatal", "kaisersesch", "glücksburg", "geyer", "beverungen", "neuenbürg", "rabenau",
               "zschopau", "wallenhorst", "nieder-olm", "planegg", "wolfstein", "rudolstadt", "nettetal", "lingen",
               "neuenhaus", "heilbad", "miltenberg", "deidesheim", "schraplau", "weimar", "eckartsberga", "lauscha",
               "eisfeld", "lebach", "gladenbach", "königslutter", "magdeburg", "rosenthal", "oberriexingen", "lollar",
               "remscheid", "merzig", "zwenkau", "helmbrechts", "bietigheim-bissingen", "schneverdingen", "bruchköbel",
               "ebersbach-neugersdorf", "laage", "mittenwalde", "penzlin", "hartha", "dahme/mark", "volkach", "peine",
               "landquart", "calbe", "mechernich", "ingelheim", "adelsheim", "oberwesel", "groß-gerau", "schwabach",
               "witten", "remagen", "aßlar", "wesel", "griesheim", "kusterdingen", "ladenburg", "viernheim",
               "neunkirchen", "rochlitz", "winnenden", "seligenstadt", "glashütte", "oberndorf", "ribnitz-damgarten",
               "ballenstedt", "höchstadt", "grünhain-beierfeld", "zwingenberg", "lychen", "schönsee", "linnich",
               "münnerstadt", "obertshausen", "schramberg", "ibbenbüren", "dingelstädt", "bopfingen", "bräunlingen",
               "crimmitschau", "metzingen", "offenburg", "gartz", "frohburg", "niddatal", "nabburg", "dornburg-camburg",
               "blankenhain", "falkenberg/elster", "berka/werra", "hallstadt", "reichenbach/o.l.", "weißenfels",
               "clingen", "bützow", "bückeburg", "mannheim", "lausanne", "parchim", "neckartenzlingen", "bernburg",
               "franzburg", "gütersloh", "gronau", "niederstotzingen", "thale", "willich", "fraubrunnen", "laufen",
               "nidderau", "görlitz", "liebenwalde", "oppenheim", "bad", "leinefelde-worbis", "neu-isenburg",
               "schwaigern", "gladbeck", "lorch", "ravenstein", "haltern", "gießen", "marktleuthen", "esens",
               "langensendelbach", "rieneck", "raunheim", "bitburg", "dorfen", "küssnacht", "st.", "dierdorf",
               "liestal", "treffurt", "pohlheim", "varel", "innsbruck", "schorndorf", "loxstedt", "herborn",
               "wahlstedt", "borgentreich", "coesfeld", "haren", "oppenau", "balve", "neu-anspach", "neckargemünd",
               "schönau", "rehau", "landsberg", "wustrow", "vacha", "eppingen", "mühlberg/elbe", "erlenbach",
               "warendorf", "neu-ulm", "waldenburg", "münchwilen", "blomberg", "illertissen", "quakenbrück",
               "grafenwöhr", "michelstadt", "karben", "lüdenscheid", "radevormwald", "steinbach", "klötze", "achim",
               "gundelsheim", "ratzeburg", "gunzenhausen", "weyhe", "schneeberg", "geseke", "plauen", "schöningen",
               "holzminden", "schriesheim", "pentling", "wittstock/dosse", "dillingen/saar", "volkmarsen", "hof",
               "eschborn", "schallstadt", "landshut", "klingenberg", "wilthen", "müncheberg", "rauenberg", "hude",
               "lichtenfels", "riesa", "wassenberg", "ulrichstein", "möckmühl", "affoltern", "burgstädt", "mosbach",
               "schwanebeck", "katzenelnbogen", "gomaringen", "buttstädt", "bredstedt", "roßwein", "espelkamp",
               "wittenberge", "wolfach", "dietzenbach", "wildenfels", "homberg", "ausgburg", "falkensee", "brühl",
               "rinteln", "neuchâtel", "freystadt", "klein-winternheim", "einbeck", "bannewitz", "mansfeld",
               "schönewalde", "weißenhorn", "finsterwalde", "halberstadt", "schönkirchen", "eggenfelden",
               "dessau-roßlau", "schwalmtal", "tangerhütte", "cremlingen", "gedern", "wolfhagen", "kirchen", "speicher",
               "kahla", "rosbach", "laatzen", "kronberg", "erftstadt", "medebach", "göttingen", "ochsenfurt",
               "stavenhagen", "quedlinburg", "oberasbach", "georgsmarienhütte", "waldeck", "röthenbach", "starnberg",
               "forchheim", "herten", "angermünde", "tirschenreuth", "schönebeck", "orlamünde", "grimma", "tessin",
               "belgern-schildau", "flörsheim", "niederkassel", "kalkar", "bogen", "hoyerswerda", "düsseldorf",
               "rosdorf", "heilbronn", "langenhagen", "laubach", "velen", "unterreichenbach", "iphofen", "visselhövede",
               "löbau", "kieselbronn", "sehnde", "waldsassen", "rockenhausen", "sprockhövel", "ichenhausen", "wissen",
               "günzburg", "schwäbisch", "freilassing", "ruhla", "füssen", "querfurt", "pößneck", "ettlingen", "kaarst",
               "alfeld", "ehrenfriedersdorf", "bexbach", "dohna", "obernburg", "neubrandenburg", "tharandt", "laucha",
               "heringen/helme", "weißenburg", "wirges", "bruchsal", "siegburg", "xanten", "altdorf", "meerbusch",
               "heinsberg", "waldheim", "werben", "sandersdorf-brehna", "baunach", "menden", "jülich", "zeitz", "aach",
               "walddorfhäslach", "schkeuditz", "sangerhausen", "offenau", "nyon", "liebenau", "ummerstadt", "cham",
               "stralsund", "schwalbach", "meißen", "idstein", "külsheim", "lambrecht", "schwandorf", "freiberg",
               "regensburg", "marktredwitz", "hilpoltstein", "leipheim", "heilsbronn", "lützen", "freyung", "grabow",
               "lommatzsch", "saalfeld/saale", "sonneberg", "zossen", "schömberg", "münsingen", "gebesee", "waldbröl",
               "altötting", "ornbau", "mönchengladbach", "selters", "eisenstadt", "allstedt", "teterow", "wilnsdorf",
               "altenholz", "baesweiler", "bellinzona", "freinsheim", "kirchentellinsfurt", "lübeck", "hirschhorn",
               "heiligenhafen", "waghäusel", "bottropp", "pettendorf", "werneuchen", "giengen", "ostfildern",
               "beilstein", "falkenstein/harz", "iserlohn", "hohenberg", "selm", "seelze", "herford", "oberwiesenthal",
               "thannhausen", "stahnsdorf", "lohmar", "dillenburg", "kämpfelbach", "rüdesheim", "vöhrenbach",
               "neukirchen", "pfullingen", "pappenheim", "lich", "elze", "mindelheim", "eislingen/fils", "lennestadt",
               "konz", "seelow", "lage", "höxter", "achern", "hilden", "engen", "rostock", "geisingen", "nordhausen",
               "vohburg", "sindelfingen", "stockelsdorf", "bassum", "furth", "egeln", "pressath", "böblingen", "rheine",
               "bönen", "dieburg", "waibstadt", "osterwieck", "eltville", "gau-algesheim", "münchen", "höchberg",
               "ochsenhausen", "moosburg", "waldmünchen", "crailsheim", "haßfurt", "heiligenhaus", "bergisch", "alfter",
               "saarbrücken", "haiterbach", "netphen", "borgholzhausen", "ransbach-baumbach", "markneukirchen",
               "herbolzheim", "drolshagen", "stadt", "zeven", "nienburg", "staßfurt", "zirndorf", "zweibrücken",
               "kaiserslautern", "brandenburg", "leonding", "isselburg", "hirschberg", "senftenberg", "meßstetten",
               "hildesheim", "remchingen", "ludwigsfelde", "neckarsteinach", "lünen", "lüchow", "oerlinghausen",
               "neulingen", "schmalkalden", "merkendorf", "maulbronn", "leisnig", "winterberg", "gerlingen", "kronach",
               "oschatz", "burladingen", "weinitzen", "werther", "leinfelden-echterdingen", "groß-umstadt", "widdern",
               "trendelburg", "solingen", "wörth", "markgröningen", "neuwied", "heubach", "schifferstadt",
               "großalmerode", "büdingen", "triptis", "voerde", "dormagen", "büdelsdorf", "gransee", "neuruppin",
               "bernau", "uster", "pleystein", "heideck", "traun", "penzberg", "hohnstein", "schwedt/oder",
               "ludwigshafen", "weida", "uelzen", "enger", "langenfeld", "schmölln", "dettelbach", "havelsee",
               "breisach", "dannenberg", "eschenbach", "zarrentin", "herbstein", "olching", "haigerloch", "hamburg",
               "neukloster", "drensteinfurt", "borchen", "grünsfeld", "herzogenaurach", "rendsburg", "herisau",
               "prévessin-moëns", "eckernförde", "frauenstein", "fulda", "hüfingen", "rehna", "radebeul", "overath",
               "putzbrunn", "ziesar", "burgau", "bramsche", "seehausen", "preußisch", "rodalben", "wassertrüdingen",
               "vellmar", "kissing", "bergen", "heldrungen", "weiden", "lauchhammer", "soltau", "reutlingen", "wemding",
               "gehrden", "koblenz", "geisenfeld", "waldershof", "lübtheen", "brück", "friedberg",
               "mühlhausen/thüringen", "nittenau", "braunsbedra", "knittlingen", "leutkirch", "haldensleben",
               "leonberg", "fröndenberg/ruhr", "stromberg", "wiesmoor", "herrenberg", "vallendar", "kalbe",
               "schillingsfürst", "urbar", "ober-olm", "putlitz", "oberhof", "ahaus", "donaustauf", "geestland",
               "karlstadt", "rheinstetten", "strehla", "langelsheim", "dahlen", "wunsiedel", "hagenbach",
               "tambach-dietharz", "wipperfürth", "oranienbaum-wörlitz", "schlüchtern", "chur", "fürstenfeldbruck",
               "steinbach-hallenberg", "herdecke", "herne", "werder", "hettingen", "preetz", "heuweiler", "schleiden",
               "lindenberg", "friedrichsdorf", "stein", "leoben", "erzhausen", "kelsterbach", "duisburg", "cuxhaven",
               "nieheim", "heppenheim", "zwickau", "arzberg", "welzow", "heide", "lößnitz", "wesenberg", "niesky",
               "wanzleben-börde", "thalheim/erzgeb.", "kastellaun", "geisenheim", "fürstenberg/havel", "wiesbaden",
               "aachen", "hohen", "kassel", "staufenberg", "hachenburg", "bönnigheim", "seifhennersdorf", "elstra",
               "berga/elster", "simbach", "friedrichshafen", "lichtenau", "hohenleuben", "haiger", "oberviechtach",
               "stadtsteinach", "marne", "löffingen", "aarau", "sandau", "hameln", "erwitte", "kaltenkirchen",
               "klingenthal", "eibenstock", "eschershausen", "holzgerlingen", "kemberg", "beckum", "spalt", "olsberg",
               "frankenthal", "friedrichroda", "frankenau", "osterburg", "delitzsch", "borna", "moringen",
               "marienmünster", "krumbach", "kindelbrück", "norden", "hochdorf", "neusäß", "wiener", "elmshorn",
               "korntal-münchingen", "dietikon", "zahna-elster", "hasbergen", "stockach", "brand-erbisdorf", "usingen",
               "datteln", "meuselwitz", "runkel", "mertesdorf", "sarnen", "oberkirch", "hornbach", "flöha", "wesseling",
               "sion", "uslar", "dillingen", "stadtbergen", "dietenheim", "schnackenburg", "nassau", "attendorn",
               "beelitz", "feldkirch", "malchin", "landstuhl", "leun", "tangermünde", "ditzingen", "überlingen",
               "kellinghusen", "kornwestheim", "bergneustadt", "soest", "taunusstein", "dommitzsch", "otterndorf",
               "erlensee", "wunstorf", "mülheim", "naumburg", "wilster", "emmelshausen", "langen", "brugg",
               "schwarzenberg/erzgeb.", "dornstetten", "herzogenrath", "krakow", "grebenstein", "loitz", "flensburg",
               "olpe", "belm", "lübbenau/spreewald", "kronshagen", "willebadessen", "römhild", "gaildorf", "vöhringen",
               "gersthofen", "lauterstein", "neckarbischofsheim", "uebigau-wahrenbrück", "weismain", "meßkirch",
               "arendsee", "ilshofen", "renchen", "osthofen", "sonthofen", "kolbermoor", "hückeswagen", "wadern",
               "renningen", "buchloe", "ulm", "geithain", "hückelhoven", "schelklingen", "hürth", "spremberg",
               "artern/unstrut", "dingolfing", "brackenheim", "schwabmünchen", "nideggen", "dortmund", "fürth",
               "jüterbog", "weißensee", "pasewalk", "boizenburg/elbe", "trochtelfingen", "eibelstadt", "crivitz",
               "hennef", "ainring", "oberweißbach/thür.", "lübbecke", "neutraubling", "manching", "boxberg",
               "burscheid", "scheßlitz", "harburg", "quierschied", "wiesensteig", "friedland", "plaue", "arnis",
               "dippoldiswalde", "appenzell", "vogtsburg", "heringen", "velden", "saalburg-ebersdorf", "rehburg-loccum",
               "hersbruck", "pausa-mühltroff", "schwarzenbach", "elchingen", "neumünster", "colditz", "penkun",
               "garbsen", "königstein", "landau", "wasserburg", "pulheim", "bocholt", "olten", "rosenfeld",
               "pfarrkirchen", "lichtenberg", "friesack", "hecklingen", "grünstadt", "ratingen", "aulendorf",
               "rorschach", "osterburken", "stadtprozelten", "rösrath", "schirgiswalde-kirschau", "frankenberg/sa.",
               "havelberg", "lügde", "rüthen", "rastenberg", "burgdorf", "glarus", "olfen", "wiesloch", "eschweiler",
               "zittau", "bernstadt", "waltershausen", "würselen", "idar-oberstein", "torgau", "johanngeorgenstadt",
               "lauta", "ludwigsstadt", "ebeleben", "warin", "niederstetten", "waltrop", "hessisch", "kempen",
               "hornberg", "groß-bieberau", "oschersleben", "wurzen", "vellberg", "tittmoning", "oberursel",
               "mainbernheim", "zehdenick", "münzenberg", "raguhn-jeßnitz", "neuffen", "kaufbeuren", "grimmen",
               "kaltennordheim", "schnaittenbach", "bismark", "seeland", "zürich", "leherheide", "hemmingen",
               "beilngries", "radolfzell", "ranis", "harsewinkel", "sigmaringen", "klütz", "woldegk", "dinklage",
               "eisingen", "bovenden", "bischofsheim", "maintal", "mölln", "burg", "tanna", "monschau", "sassenberg",
               "oelsnitz/erzgeb.", "bonndorf", "ravensburg", "burghausen", "birkenfeld", "nauen", "schlitz", "dahn",
               "tuttlingen", "sendenhorst", "ronneburg", "waldkappel", "dornstadt", "weißenberg", "solothurn",
               "emsdetten", "staufen", "hégenheim", "märkisch", "rödermark", "velbert", "karlsruhe", "gelnhausen",
               "öhringen", "tribsees", "balingen", "bonn", "hainichen", "sternberg", "bremervörde", "böhlen",
               "greifswald", "altena", "sulzbach-rosenberg", "bielefeld", "versmold", "neresheim", "brilon",
               "zierenberg", "gräfenthal", "neudenau", "telgte", "calw", "adorf/vogtl.", "braunschweig", "sankt",
               "dielsdorf", "anklam", "weinsberg", "furtwangen", "merseburg", "niedenstein", "steinau",
               "friedrichsthal", "kirchheimbolanden", "taucha", "wiefelstede", "schwetzingen", "neubukow",
               "plettenberg", "kehl", "völs", "marktsteft", "rerik", "kühlungsborn", "himmelreich", "golßen",
               "rödental", "kuppenheim", "grünberg", "wolgast", "lassan", "troisdorf", "edingen-neckarhausen",
               "blankenburg", "erbendorf", "eppstein", "radeburg", "großschirma", "ochtrup", "mitterteich", "themar",
               "wildeshausen", "kandern", "tönisvorst", "pinneberg", "ansfelden", "altensteig", "weil",
               "lichtenstein/sa.", "lauterbach", "elterlein", "deggendorf", "stadtroda", "sulza", "altenburg",
               "sassnitz", "elzach", "salzburg", "zug", "geldern", "schwarzheide", "schwarzenbek", "tettnang",
               "springe", "westerburg", "glauchau", "lohr", "osterfeld", "nienburg/weser", "ennepetal", "beerfelden",
               "rottweil", "sulzbach/saar", "unterhaching", "itzehoe", "sömmerda", "viersen", "aschersleben",
               "altenkirchen", "wurzbach", "ortrand", "weilheim", "breckerfeld", "sontra", "kroppenstedt",
               "weißwasser/o.l.", "owen", "arnstein", "osterholz-scharmbeck", "liebstadt", "kelheim", "wangen",
               "bretten", "ginsheim-gustavsburg", "ulmen", "gefrees", "lugano", "hirschau", "pfaffenhofen", "villach",
               "veringenstadt", "miesbach", "ampass", "coburg", "gotha", "krefeld", "mügeln", "penig", "glinde",
               "arnstadt", "olbernhau", "gadebusch", "laufamholzer", "oberschleißheim", "hallenberg", "rothenburg",
               "euskirchen", "treuen", "freyburg", "lorsch", "wegeleben", "lindenfels", "heimbach", "guben", "grafing",
               "bodenheim", "ansbach", "vohenstrauß", "rüsselsheim", "geringswalde", "salzkotten", "waldkirch",
               "ketzin/havel", "elsfleth", "kappeln", "stadthagen", "monheim", "joachimsthal", "diepholz", "lahnstein",
               "stolpen", "wyk", "bargteheide", "detmold", "eningen", "gummersbach", "jüchen", "neuerburg", "celle",
               "lebus", "regen", "gräfenberg", "gemünden", "storkow", "fridingen", "aalen", "rothenburg/o.l.",
               "garding", "nürnberg", "wachtberg", "aub", "wenzenbach", "aurich", "mellrichstadt", "teuchern", "halle",
               "hatzfeld", "windsbach", "weikersheim", "dietfurt", "niebüll", "petershagen", "pfullendorf",
               "meinerzhagen", "meerane", "gleichen", "cloppenburg", "goldberg", "wendlingen", "kempten", "wittmund",
               "ziegenrück", "korschenbroich", "halver", "unterschleißheim", "schwentinental", "worms", "östringen",
               "linz", "papenburg", "doberlug-kirchhain", "dossenheim", "herbrechtingen", "bochum", "hartenstein",
               "baumholder", "straubing", "wernau", "steinfurt", "pfäffikon", "linden", "kirchhain", "meisenheim",
               "meldorf", "randersacker", "wettin-löbejün", "hungen", "nußloch", "bentwisch", "baiersdorf",
               "creglingen", "maxhütte-haidhof", "eilenburg", "nossen", "bülach", "usedom", "borkum",
               "stollberg/erzgeb.", "hayingen", "kamp-lintfort", "offenbach", "konolfingen", "forchtenberg", "basel",
               "eisenberg", "wilsdruff", "pocking", "romrod", "salzgitter", "schalkau", "markranstädt", "fritzlar",
               "neuenburg", "gardelegen", "stuttgart", "ahrensburg", "vreden", "auerbach", "herrnhut", "obermoschel",
               "andernach", "bärnau", "güsten", "königsee-rottenbach", "schopfheim", "singen", "eberbach", "kirn",
               "edenkoben", "riedlingen", "roßdorf", "kölleda", "eggesin", "putbus", "gengenbach", "jöhstadt",
               "müllrose", "ottendorf", "forst", "lehrte", "traunstein", "ober-ramstadt", "freital", "sissach",
               "norderstedt", "waldshut-tiengen", "rennerod", "walldürn", "ronnenberg", "wachenheim",
               "oer-erkenschwick", "grafenau", "sebnitz", "nastätten", "erkelenz", "recklinghausen", "königsberg",
               "spaichingen", "mettmann", "teublitz", "munderkingen", "bersenbrück", "burgkunstadt", "fuldatal",
               "neusalza-spremberg", "waldkirchen", "kohren-sahlis", "herdorf", "heidelberg", "wallenfels", "lübz",
               "altenberg", "rottenburg", "augsburg", "buckow", "oederan", "roßleben", "rötha", "kitzscher", "northeim",
               "altentreptow", "limburg", "bensheim", "keltern", "gaggenau", "uhingen", "rötz", "mühlheim", "coswig",
               "pockau-lengefeld", "esslingen", "oranienburg", "gehren", "rheinsberg", "villingen-schwenningen",
               "rothenfels", "roth", "ilsenburg", "garz/rügen", "hattersheim", "röttingen", "kiel", "braubach", "löhne",
               "gescher", "saarlouis", "linkenheim-hochstetten", "dorsten", "mayen", "buxtehude", "schlotheim", "stade",
               "buttelstedt", "schlüsselfeld", "titisee-neustadt", "oderberg", "dachau", "wetzlar", "göppingen",
               "bremen", "ingelfingen", "nordhorn", "manderscheid", "barntrup", "seeheim-jugenheim", "borken",
               "königsbrunn", "schöneck/vogtl.", "großenehrich", "gröningen", "simmern/hunsrück", "neubulach",
               "bergheim", "oelsnitz/vogtl.", "arlesheim", "schrozberg", "filderstadt", "oberkochen", "lübben",
               "stadtilm", "betzenstein", "neuenstein", "völklingen", "mirow", "stadtallendorf", "klagenfurt",
               "gersfeld", "blumberg", "nierstein", "lauda-königshofen", "roding", "kelkheim", "pirka", "remda-teichel",
               "oelde", "lengenfeld", "kröpelin", "laichingen", "rheinberg", "zofingen", "emmendingen", "leverkusen",
               "kirchberg", "gifhorn", "erfurt", "heikendorf", "eichstätt", "wiehl", "laupen", "arnsberg", "husum",
               "lenzen", "kemnath", "bodenwerder", "ilmenau", "markkleeberg", "grebenau", "barmstedt", "speyer",
               "ruhland", "porta", "wittichenau", "prüm", "schloß", "traben-trarbach", "schwarzenborn", "freising",
               "kelbra", "kierspe", "krautheim", "peitz", "lenting", "weiterstadt", "unkel", "lauchheim", "delbrück",
               "selb", "hardegsen", "norderney", "gefell", "jerichow", "ottweiler", "rhede", "rodewisch",
               "zeulenroda-triebes", "wolfsburg", "lindau", "hann."}


class WordFinder():
    def __init__(self, min_occurrence=10, window=15, from_corpus=False):
        self.min_occurrence = min_occurrence
        self.window = window

        # map words to integers (more memory efficient and faster)
        self.word2int_count = count()
        self.word2int = defaultdict(self.word2int_count.__next__)

        # map city names also to ints
        self.city2int_count = count()
        self.city2int = defaultdict(self.city2int_count.__next__)

        self.stemmer = SnowballStemmer('german')
        self.stopwords = set(stopwords.words('german')).union(STOP_CITIES)
        self.stems = defaultdict(lambda: defaultdict(int))

        self.cores = multiprocessing.cpu_count()

        if from_corpus:
            print("loading spacy", file=sys.stderr, flush=True)
            self.nlp = spacy.load('de', parser=False, tagger=True, entity=False)
            print("done...", file=sys.stderr, flush=True)

    def clean(self, message):
        """
        tokenize and clean a jodel message
        :param message:
        :return:
        """
        tokens = [token.text.lower() for token in self.nlp(message) if
                  token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'X'}
                  and token.text.lower() not in self.stopwords
                  and token.ent_type_ == '']
        stems = [self.stemmer.stem(token) for token in tokens]
        for (token, stem) in zip(tokens, stems):
            self.stems[stem][token] += 1
        return [self.word2int[stem] for stem in stems]

    def join_collocations(self, element):
        """
        join collocations in element (=list)
        :param element: a list of ints
        :param collocations: a set of the found collocations
        :return: joint version of input, list of strings
        """
        result = []
        is_collocation = False
        current_chain = []
        for i, w in enumerate(element):
            if i < len(element) - 1 and (w, element[i + 1]) in self.collocations:
                if current_chain == []:
                    current_chain = [w, element[i + 1]]
                else:
                    current_chain.append(element[i + 1])
                is_collocation = True
            else:
                if is_collocation:
                    new_word = '_'.join([self.int2word[ce] for ce in current_chain])
                    # new_word = '_'.join([ce for ce in current_chain])
                    self.stems[new_word][new_word] += 1
                    # result.append(self.word2int[new_word])
                    result.append(new_word)
                    current_chain = []
                else:
                    # result.append(w)
                    result.append(self.int2word[w])
                is_collocation = False
        return result

    def collect(self, input_file, limit=None, save=None):
        """
        corllect corpus from Jodel JSON data
        :param input_file:
        :param limit:
        :param save:
        :return:
        """
        corpus = []
        words = []
        labels = []

        identities = {
            'Basel-Stadt': 'Basel',
            'Brunswick': 'Braunschweig',
            'Cologne': 'Köln',
            'Frankfurt': 'Frankfurt am Main',
            'Freiburg': 'Freiburg im Breisgau',
            'Fribourg-en-Brisgau': 'Freiburg im Breisgau',
            'Geneva': 'Genf',
            'Genève': 'Genf',
            'Hanover': 'Hannover',
            'Klagenfurt am Wörthersee': 'Klagenfurt',
            'Munich': 'München',
            'Nuremberg': 'Nürnberg',
            'Ouest lausannois': 'Lausanne',
            'Sankt Pölten': 'St. Pölten',
            'Sankt Gallen': 'St. Gallen',
            'Salzburg-Umgebung': 'Salzburg',
            'Vienna': 'Wien',
            'Zurich': 'Zürich'
        }

        self.city_frequency = defaultdict(int)

        # iterate over the data
        with open(input_file, encoding='utf-8', errors='ignore') as f:
            for line_no, line in enumerate(islice(f, None)):
                if line_no > 0:
                    if line_no % 10000 == 0:
                        print("%s" % (line_no), file=sys.stderr, flush=True)
                    elif line_no % 500 == 0:
                        print('.', file=sys.stderr, end=' ', flush=True)

                try:
                    jodel = json.loads(line)
                except ValueError:
                    print(line)
                msg = jodel.get('message', None)
                location = jodel.get('location', None)

                # skip empty jodels
                if msg is None or location is None:
                    continue

                city = location.get('name', None)

                if city == 'Jodel Team' or city is None:
                    continue

                # correct city names
                city = identities.get(city, city)

                self.city_frequency[city] += 1

                # collect all the data and transform it
                data = [self.clean(msg)]
                data.extend([self.clean(child.get('message', [])) for child in jodel.get('children', [])])

                # one instance for each jodel
                # corpus.extend(data)
                # labels.extend([city] * len(data))

                # one instance for each conversation
                corpus.append([word for message in data for word in message])
                labels.append(city)

                words.extend([word for message in data for word in message])

                if limit is not None and line_no == limit:
                    break

        assert len(labels) == len(
            corpus), "umm, the number of labels (%s) and the number of instances (%s) is not the same" % (
            len(labels), len(corpus))

        self.int2word = {i: max(self.stems[w].keys(), key=(lambda key: self.stems[w][key])) for w, i in
                         self.word2int.items()}
        # find collocations
        print('\nlooking for collocations', file=sys.stderr, flush=True)
        finder = BigramCollocationFinder.from_words(words)
        bgm = BigramAssocMeasures()
        collocations = [b for b, f in finder.score_ngrams(bgm.mi_like) if f > 1.0]
        self.collocations = set(collocations)

        print('\ncreating corpus', file=sys.stderr, flush=True)
        if save is not None:
            self.corpus = []
            with open('%s.corpus' % save, 'w', encoding='utf-8') as save_corpus:
                for doc, tag in zip(corpus, labels):
                    words = self.join_collocations(doc)
                    tags = [tag]
                    self.corpus.append(TaggedDocument(words, tags=tags))
                    save_corpus.write('%s\n' % json.dumps({'words': words, 'tags': tags}))
            print('\ncorpus saved as %s' % save, file=sys.stderr, flush=True)

            with open('%s.citycounts' % save, 'w', encoding='utf-8') as save_counts:
                json.dump(dict(self.city_frequency), save_counts)

        else:
            self.corpus = [TaggedDocument(self.join_collocations(doc), tags=[tag]) for doc, tag in zip(corpus, labels)]

        print('\n%s instances' % len(self.corpus), file=sys.stderr, flush=True)

        # update mappings
        self.int2word = {i: max(self.stems[w].keys(), key=(lambda key: self.stems[w][key])) for w, i in
                         self.word2int.items()}
        print("Found %s collocations" % (len(collocations)), file=sys.stderr, flush=True)
        for (w1, w2) in collocations[:10]:
            print('\t', self.int2word[w1], self.int2word[w2], file=sys.stderr, flush=True)

    def load_corpus(self, corpus_name):
        """
        load a pre-processed corpus
        :param corpus_name:
        :return:
        """
        self.corpus = []
        self.city_frequency = defaultdict(int)
        print("loading corpus...", end='\t', file=sys.stderr, flush=True)
        with open(corpus_name, encoding='utf-8') as corpus_file:
            for line in corpus_file:
                instance = json.loads(line.strip())
                words = instance['words']
                tags = instance['tags']
                self.city_frequency[tags[0]] += 1
                self.corpus.append(TaggedDocument(words, tags=tags))
        print("done", file=sys.stderr, flush=True)

    def fit(self, save=None):
        '''
        train d2v
        :return:
        '''
        print('\nbuilding model', file=sys.stderr, flush=True)
        self.model = Doc2Vec(size=300, window=self.window, min_count=self.min_occurrence, negative=5, hs=0,
                             workers=self.cores, iter=10, sample=0.00001, dm=0, dbow_words=1)

        self.model.build_vocab(self.corpus)

        print('\ntraining model', file=sys.stderr, flush=True)
        # account for the different signatures in Python 3.5 and 3.4
        try:
            self.model.train(self.corpus, total_examples=self.model.corpus_count, epochs=self.model.iter)
        except TypeError:
            self.model.train(self.corpus)
        print('... done!', file=sys.stderr, flush=True)

        # remove unneeded memory stuff
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

        if save is not None:
            self.model.save('%s.model' % save)
            print('\nmodel saved as %s.model' % save, file=sys.stderr, flush=True)

            # for d in sorted(self.model.docvecs.doctags.keys()):
            #     if d.startswith('ID'):
            #         continue
            #     print(d, self.city_frequency[d])
            #     print(self.model.docvecs.most_similar(d, topn=10))
            #     print(self.model.most_similar(self.model.docvecs[[d]]))
            #     print()


wf = WordFinder(min_occurrence=args.min_tf, from_corpus=args.load_corpus is None)
if args.load_corpus:
    wf.load_corpus(args.load_corpus)
else:
    wf.collect(args.input, limit=args.limit, save=args.save_corpus)
wf.fit(save=args.save_model)
