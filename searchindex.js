Search.setIndex({"docnames": ["api", "graphic", "index", "introduction", "license", "model", "scene/desired_velocities", "scene/geometry", "scene/index", "scene/obstacles", "scene/walkers", "solver"], "filenames": ["api.rst", "graphic.rst", "index.rst", "introduction.rst", "license.rst", "model.rst", "scene/desired_velocities.rst", "scene/geometry.rst", "scene/index.rst", "scene/obstacles.rst", "scene/walkers.rst", "solver.rst"], "titles": ["API Reference", "Plotting the walk", "Welcome to MockWalkers\u2019s documentation!", "Introduction", "License", "Walking model", "Desired velocities", "Geometry", "Composing the scene", "Obstacles", "Walkers", "Solving the walk"], "terms": {"section": [0, 1, 4, 6, 7, 9, 10], "under": [0, 1, 2, 4, 6, 7, 9, 10], "construct": [0, 1, 6, 7, 9, 10], "thi": [2, 3, 4, 11], "project": [2, 3], "i": [2, 3, 4, 5, 11], "develop": [2, 3, 4], "introduct": 2, "start": [2, 4], "begin": 2, "nutshel": 2, "A": [2, 4, 5], "big": 2, "shout": 2, "out": [2, 4, 11], "walk": [2, 3], "model": [2, 3, 4], "solv": 2, "plot": 2, "compos": 2, "scene": 2, "walker": [2, 5, 8, 11], "desir": [2, 5, 8, 11], "veloc": [2, 5, 8, 11], "obstacl": [2, 5, 8, 11], "geometri": [2, 8, 11], "api": 2, "refer": [2, 4], "licens": 2, "origin": [3, 4], "our": [3, 4, 5, 11], "final": [3, 4], "dure": 3, "9th": 3, "workshop": 3, "collabor": 3, "scientif": 3, "softwar": [3, 4], "manag": 3, "open": 3, "sourc": [3, 4], "packag": [3, 4], "held": 3, "intern": 3, "centr": 3, "theoret": 3, "physic": [3, 4], "ictp": 3, "picturesqu": 3, "citi": 3, "triest": 3, "itali": 3, "put": 3, "simpli": 3, "librari": [3, 4, 11], "design": [3, 4], "simul": 3, "human": 3, "behavior": 3, "custom": [3, 4], "environ": 3, "approach": [3, 5, 11], "draw": 3, "inspir": 3, "from": [3, 4, 5, 11], "helb": 3, "molnar": 3, "": [3, 4, 11], "social": 3, "forc": [3, 4, 5], "pedestrian": 3, "dynam": [3, 4, 5], "phy": 3, "rev": 3, "e": [3, 4, 5], "1995": 3, "we": [3, 4, 11], "d": [3, 4], "like": [3, 4, 5, 11], "extend": [3, 4], "heartfelt": 3, "appreci": 3, "organ": [3, 4], "except": [3, 4], "effort": [3, 4], "creat": 3, "an": [3, 4], "enlighten": 3, "thoroughli": 3, "enjoy": 3, "experi": 3, "gnu": 4, "gener": 4, "public": 4, "version": 4, "3": 4, "29": 4, "june": 4, "2007": 4, "copyright": 4, "c": 4, "free": 4, "foundat": 4, "inc": 4, "http": 4, "fsf": 4, "org": 4, "everyon": 4, "permit": 4, "copi": 4, "distribut": 4, "verbatim": 4, "document": 4, "chang": 4, "allow": [4, 11], "preambl": 4, "The": [4, 5, 11], "copyleft": 4, "other": 4, "kind": 4, "work": 4, "most": 4, "practic": 4, "ar": [4, 5, 11], "take": 4, "awai": 4, "your": 4, "freedom": 4, "share": 4, "By": 4, "contrast": 4, "intend": [4, 5], "guarante": 4, "all": 4, "program": 4, "make": 4, "sure": 4, "remain": 4, "its": 4, "user": 4, "us": [4, 11], "appli": 4, "also": 4, "ani": [4, 11], "releas": 4, "wai": 4, "author": 4, "you": [4, 11], "can": 4, "too": 4, "when": [4, 5], "speak": 4, "price": 4, "have": 4, "charg": 4, "them": 4, "wish": 4, "receiv": 4, "code": [4, 11], "get": 4, "want": 4, "piec": 4, "new": 4, "know": 4, "do": [4, 11], "thing": 4, "To": [4, 11], "protect": 4, "right": [4, 5], "need": 4, "prevent": 4, "deni": 4, "ask": 4, "surrend": 4, "therefor": 4, "certain": [4, 5], "respons": 4, "modifi": 4, "respect": 4, "For": [4, 11], "exampl": 4, "whether": 4, "grati": 4, "fee": 4, "must": 4, "pass": 4, "recipi": 4, "same": 4, "thei": 4, "And": 4, "show": 4, "term": 4, "so": 4, "gpl": 4, "two": 4, "step": [4, 11], "1": [4, 5, 11], "assert": 4, "2": [4, 5], "offer": 4, "give": 4, "legal": 4, "permiss": 4, "clearli": 4, "explain": [4, 11], "warranti": 4, "both": 4, "sake": 4, "requir": 4, "mark": 4, "problem": [4, 11], "attribut": 4, "erron": 4, "previou": 4, "some": 4, "devic": 4, "access": 4, "instal": 4, "run": 4, "insid": 4, "although": 4, "manufactur": 4, "fundament": 4, "incompat": 4, "aim": [4, 11], "systemat": 4, "pattern": 4, "abus": 4, "occur": 4, "area": 4, "product": 4, "individu": [4, 5], "which": [4, 5, 11], "precis": 4, "where": [4, 5], "unaccept": 4, "prohibit": 4, "those": 4, "If": 4, "aris": 4, "substanti": 4, "domain": 4, "stand": 4, "readi": 4, "provis": 4, "futur": 4, "everi": 4, "threaten": 4, "constantli": 4, "patent": 4, "state": [4, 11], "should": 4, "restrict": 4, "purpos": 4, "comput": [4, 11], "avoid": 4, "special": 4, "danger": 4, "could": 4, "effect": 4, "proprietari": 4, "assur": 4, "cannot": 4, "render": 4, "non": 4, "condit": 4, "modif": 4, "follow": 4, "AND": 4, "definit": 4, "mean": 4, "law": 4, "semiconductor": 4, "mask": 4, "each": [4, 11], "license": 4, "address": 4, "mai": 4, "adapt": 4, "part": 4, "fashion": 4, "than": 4, "exact": 4, "result": 4, "call": [4, 11], "earlier": 4, "base": 4, "cover": 4, "either": 4, "unmodifi": 4, "propag": 4, "anyth": 4, "without": 4, "would": 4, "directli": 4, "secondarili": 4, "liabl": 4, "infring": 4, "applic": 4, "execut": 4, "privat": 4, "includ": 4, "avail": 4, "countri": 4, "activ": 4, "well": 4, "convei": 4, "enabl": 4, "parti": 4, "mere": 4, "interact": [4, 5], "through": 4, "network": 4, "transfer": 4, "interfac": 4, "displai": 4, "appropri": 4, "notic": 4, "extent": 4, "conveni": 4, "promin": 4, "visibl": 4, "featur": 4, "tell": 4, "provid": 4, "how": 4, "view": [4, 5], "present": 4, "list": 4, "command": 4, "option": 4, "menu": 4, "item": 4, "meet": 4, "criterion": 4, "prefer": 4, "form": [4, 5], "object": 4, "standard": 4, "offici": 4, "defin": [4, 11], "recogn": 4, "bodi": 4, "case": 4, "specifi": 4, "particular": 4, "languag": 4, "one": [4, 11], "wide": 4, "among": 4, "system": 4, "whole": 4, "normal": [4, 5], "major": 4, "compon": [4, 5], "b": [4, 5], "serv": 4, "onli": [4, 5], "implement": 4, "context": 4, "essenti": 4, "kernel": [4, 5], "window": 4, "specif": [4, 11], "oper": 4, "compil": 4, "produc": 4, "interpret": 4, "correspond": 4, "script": 4, "control": 4, "howev": 4, "doe": 4, "tool": 4, "perform": [4, 11], "file": 4, "associ": 4, "link": 4, "subprogram": 4, "intim": 4, "data": 4, "commun": 4, "flow": 4, "between": [4, 5], "regener": 4, "automat": 4, "basic": 4, "grant": 4, "irrevoc": 4, "met": 4, "explicitli": 4, "affirm": 4, "unlimit": 4, "output": 4, "given": 4, "content": 4, "constitut": 4, "acknowledg": 4, "fair": 4, "equival": 4, "long": 4, "otherwis": [4, 5], "sole": 4, "exclus": 4, "facil": 4, "compli": 4, "materi": 4, "thu": 4, "behalf": 4, "direct": 4, "outsid": 4, "relationship": 4, "circumst": 4, "below": 4, "sublicens": 4, "10": 4, "unnecessari": 4, "anti": 4, "circumvent": 4, "No": 4, "shall": 4, "deem": 4, "technolog": 4, "measur": 4, "fulfil": 4, "oblig": 4, "articl": 4, "11": 4, "wipo": 4, "treati": 4, "adopt": 4, "20": 4, "decemb": 4, "1996": 4, "similar": [4, 11], "waiv": 4, "power": 4, "forbid": 4, "exercis": 4, "disclaim": 4, "intent": 4, "limit": 4, "enforc": 4, "against": 4, "third": 4, "medium": 4, "conspicu": 4, "publish": 4, "keep": 4, "intact": 4, "ad": 4, "accord": 4, "7": 4, "absenc": 4, "along": 4, "support": 4, "4": 4, "carri": [4, 11], "relev": 4, "date": 4, "entir": 4, "anyon": 4, "who": 4, "come": [4, 5], "possess": 4, "addit": 4, "regardless": 4, "invalid": 4, "separ": 4, "ha": 4, "independ": 4, "natur": 4, "extens": 4, "combin": 4, "larger": 4, "volum": 4, "storag": 4, "aggreg": 4, "beyond": 4, "what": [4, 11], "inclus": 4, "caus": [4, 5], "5": 4, "machin": 4, "readabl": 4, "embodi": [4, 5], "accompani": 4, "fix": 4, "durabl": 4, "customarili": 4, "interchang": 4, "written": 4, "valid": 4, "least": 4, "three": 4, "year": 4, "spare": 4, "more": 4, "reason": 4, "cost": 4, "server": 4, "altern": 4, "occasion": 4, "noncommerci": 4, "subsect": 4, "6b": 4, "place": 4, "further": 4, "differ": 4, "maintain": 4, "clear": 4, "next": 4, "sai": [4, 11], "find": [4, 11], "host": 4, "ensur": 4, "satisfi": 4, "peer": 4, "transmiss": 4, "inform": 4, "being": 4, "6d": 4, "portion": 4, "whose": 4, "exclud": 4, "consum": 4, "tangibl": 4, "person": 4, "properti": 4, "famili": 4, "household": 4, "sold": 4, "incorpor": 4, "dwell": 4, "In": [4, 11], "determin": [4, 5, 11], "doubt": 4, "resolv": 4, "favor": 4, "coverag": 4, "typic": [4, 5], "common": 4, "class": [4, 11], "statu": 4, "actual": 4, "expect": 4, "commerci": 4, "industri": 4, "unless": 4, "repres": [4, 5, 11], "signific": 4, "mode": 4, "method": [4, 11], "procedur": 4, "kei": [4, 5], "suffic": 4, "continu": 4, "function": 4, "interf": 4, "becaus": 4, "been": 4, "made": 4, "transact": 4, "perpetu": 4, "character": 4, "But": 4, "neither": 4, "nor": 4, "retain": 4, "abil": 4, "rom": 4, "servic": 4, "updat": 4, "itself": 4, "advers": 4, "affect": 4, "violat": 4, "rule": 4, "protocol": 4, "across": 4, "format": 4, "publicli": 4, "password": 4, "unpack": 4, "read": 4, "supplement": 4, "treat": 4, "though": 4, "were": 4, "govern": 4, "regard": 4, "remov": 4, "own": 4, "notwithstand": 4, "add": 4, "holder": 4, "liabil": 4, "15": 4, "16": 4, "preserv": 4, "contain": 4, "misrepresent": 4, "name": 4, "licensor": 4, "declin": 4, "trademark": 4, "trade": 4, "f": [4, 5], "indemnif": 4, "contractu": 4, "assumpt": 4, "impos": 4, "consid": 4, "within": [4, 5], "relicens": 4, "surviv": 4, "statement": 4, "indic": 4, "abov": 4, "termin": 4, "expressli": 4, "attempt": 4, "void": 4, "paragraph": 4, "ceas": 4, "reinstat": 4, "provision": 4, "until": 4, "perman": 4, "fail": 4, "notifi": 4, "prior": 4, "60": 4, "dai": 4, "after": 4, "cessat": 4, "moreov": 4, "first": 4, "time": [4, 5, 11], "cure": 4, "30": 4, "receipt": 4, "qualifi": 4, "accept": 4, "Not": 4, "order": 4, "ancillari": 4, "consequ": 4, "likewis": 4, "noth": 4, "These": 4, "action": 4, "downstream": 4, "subject": 4, "complianc": 4, "entiti": 4, "asset": 4, "subdivid": 4, "merg": 4, "whatev": 4, "predecessor": 4, "interest": 4, "had": 4, "plu": 4, "royalti": 4, "initi": 4, "litig": 4, "cross": 4, "claim": 4, "counterclaim": 4, "lawsuit": 4, "alleg": 4, "sell": 4, "sale": 4, "import": [4, 11], "contributor": 4, "alreadi": 4, "acquir": 4, "hereaft": 4, "manner": 4, "consist": 4, "worldwid": 4, "express": [4, 11], "agreement": 4, "commit": 4, "denomin": 4, "coven": 4, "sue": 4, "knowingli": 4, "reli": [4, 5], "readili": 4, "arrang": 4, "depriv": 4, "yourself": 4, "benefit": 4, "knowledg": 4, "identifi": 4, "believ": 4, "pursuant": 4, "connect": 4, "singl": 4, "procur": 4, "convey": 4, "discriminatori": 4, "scope": 4, "busi": 4, "payment": 4, "primarili": 4, "enter": 4, "wa": 4, "28": 4, "march": 4, "constru": 4, "impli": 4, "defens": 4, "court": 4, "contradict": 4, "excus": 4, "simultan": 4, "pertin": 4, "agre": 4, "collect": 4, "whom": 4, "refrain": 4, "affero": 4, "13": 4, "concern": 4, "revis": 4, "Such": 4, "spirit": 4, "detail": [4, 11], "distinguish": 4, "number": 4, "later": [4, 11], "choos": 4, "ever": 4, "proxi": 4, "decid": 4, "THERE": 4, "NO": 4, "FOR": 4, "THE": 4, "TO": 4, "BY": 4, "IN": 4, "write": 4, "OR": 4, "AS": 4, "OF": 4, "BUT": 4, "NOT": 4, "merchant": 4, "fit": 4, "risk": 4, "qualiti": 4, "WITH": 4, "prove": 4, "defect": 4, "assum": 4, "necessari": 4, "repair": 4, "correct": 4, "event": 4, "WILL": 4, "BE": 4, "damag": 4, "incident": 4, "consequenti": 4, "inabl": 4, "loss": 4, "BEING": 4, "inaccur": 4, "sustain": 4, "failur": 4, "even": 4, "IF": 4, "SUCH": 4, "advis": 4, "possibl": 4, "local": 4, "review": 4, "close": 4, "approxim": 4, "absolut": 4, "waiver": 4, "civil": 4, "return": 4, "end": 4, "greatest": 4, "best": 4, "achiev": 4, "redistribut": 4, "attach": 4, "It": [4, 5], "safest": 4, "line": 4, "pointer": 4, "full": 4, "found": 4, "brief": 4, "idea": 4, "2022": 4, "abdulganiyu": 4, "jimoh": 4, "hope": 4, "see": 4, "www": 4, "contact": 4, "electron": 4, "paper": 4, "mail": 4, "short": 4, "type": 4, "w": 4, "welcom": 4, "hypothet": 4, "Of": 4, "cours": [4, 11], "might": [4, 11], "gui": 4, "about": 4, "box": 4, "employ": 4, "programm": 4, "school": 4, "sign": 4, "subroutin": 4, "lesser": 4, "instead": 4, "pleas": 4, "why": 4, "lgpl": 4, "html": 4, "let": 5, "x_i": 5, "left": 5, "t": [5, 11], "y_i": 5, "posit": [5, 11], "0": 5, "crowd": 5, "n": [5, 11], "set": [5, 11], "newton": 5, "ordinari": 5, "differenti": 5, "equat": [5, 11], "od": [5, 11], "describ": [5, 11], "ddot": [5, 11], "x": [5, 11], "_i": 5, "dot": [5, 11], "sum_": 5, "neq": 5, "j": 5, "k": 5, "x_j": 5, "2d": 5, "plane": 5, "here": [5, 11], "regul": 5, "propuls": 5, "frac": 5, "v_d": 5, "tau": 5, "field": 5, "relax": 5, "pairwis": 5, "exp": 5, "r": 5, "theta": 5, "repuls": 5, "plai": 5, "cone": 5, "paramet": 5, "radiu": 5, "zero": 5, "angl": 5, "exce": 5, "threshold": 5, "g": 5, "80": 5, "degre": 5, "account": 5, "d_": 5, "r_k": 5, "prime": 5, "vec": 5, "n_k": 5, "distanc": 5, "vector": 5, "point": [5, 11], "toward": 5, "scale": 5, "imperm": 5, "solut": 11, "emploi": 11, "straightforward": 11, "euler": 11, "scheme": 11, "discret": 11, "delta": 11, "evolut": 11, "x_": 11, "_": 11, "notat": 11, "acceler": 11, "calcul": 11, "solver": 11, "simplifi": 11, "process": 11, "significantli": 11, "instanc": 11, "second": 11, "look": 11, "mockwalk": 11, "mckw": 11, "delta_t": 11, "vd_calc": 11, "while": 11, "current_tim": 11, "iter": 11, "someth": 11, "variou": 11, "element": 11, "crucial": 11, "note": 11, "advanc": 11, "numer": 11, "obtain": 11}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"api": 0, "refer": 0, "plot": 1, "walk": [1, 5, 11], "welcom": 2, "mockwalk": [2, 3], "": 2, "document": 2, "introduct": 3, "start": 3, "begin": 3, "nutshel": 3, "A": 3, "big": 3, "shout": 3, "out": 3, "licens": 4, "model": 5, "desir": 6, "veloc": 6, "geometri": 7, "compos": 8, "scene": 8, "obstacl": 9, "walker": 10, "solv": 11}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 58}, "alltitles": {"API Reference": [[0, "api-reference"]], "Plotting the walk": [[1, "plotting-the-walk"]], "Welcome to MockWalkers\u2019s documentation!": [[2, "welcome-to-mockwalkers-s-documentation"]], "Introduction": [[3, "introduction"]], "Starting at the Beginning": [[3, "starting-at-the-beginning"]], "MockWalkers in a nutshell": [[3, "mockwalkers-in-a-nutshell"]], "A Big Shout-Out": [[3, "a-big-shout-out"]], "License": [[4, "license"]], "Walking model": [[5, "walking-model"]], "Desired velocities": [[6, "desired-velocities"]], "Geometry": [[7, "geometry"]], "Composing the scene": [[8, "composing-the-scene"]], "Obstacles": [[9, "obstacles"]], "Walkers": [[10, "walkers"]], "Solving the walk": [[11, "solving-the-walk"]]}, "indexentries": {}})