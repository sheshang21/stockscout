import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Indian Stock Scout - Ultra-Strict Scanner", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:700;color:#1f77b4;text-align:center;margin-bottom:1rem}
.sub-header{font-size:1.5rem;font-weight:600;color:#333;margin:1rem 0}
.metric-card{background:#f8f9fb;padding:0.8rem;border-radius:8px;border-left:4px solid #1f77b4;margin:0.5rem 0}
.stDataFrame{font-size:0.9rem}
div[data-testid="stDataFrame"] > div{background:#f8f9fb}
.price-up{color:#00c853;font-weight:bold}
.price-down{color:#ff1744;font-weight:bold}
.price-neutral{color:#666}
</style>""", unsafe_allow_html=True)

# Comprehensive Stock Universe - 200+ NSE Stocks
NSE_STOCKS = [
    "20MICRONS", "21STCENMGM", "360ONE", "3IINFOLTD", "3MINDIA", 
                "3PLAND", "5PAISA", "63MOONS", "A2ZINFRA", "AAATECH", 
                "AADHARHFC", "AAKASH", "AAREYDRUGS", "AARON", "AARTECH", 
                "AARTIDRUGS", "AARTIIND", "AARTIPHARM", "AARTISURF", "AARVI", 
                "AAVAS", "ABAN", "ABB", "ABBOTINDIA", "ABCAPITAL", 
                "ABCOTS", "ABDL", "ABFRL", "ABINFRA", "ABLBL", 
                "ABMINTLLTD", "ABREL", "ABSLAMC", "ACC", "ACCELYA", 
                "ACCURACY", "ACE", "ACEINTEG", "ACI", "ACL", 
                "ACMESOLAR", "ACUTAAS", "ADANIENSOL", "ADANIENT", "ADANIGREEN", 
                "ADANIPORTS", "ADANIPOWER", "ADFFOODS", "ADL", "ADOR", 
                "ADROITINFO", "ADSL", "ADVANCE", "ADVANIHOTR", "ADVENTHTL", 
                "ADVENZYMES", "AEGISLOG", "AEGISVOPAK", "AEROENTER", "AEROFLEX", 
                "AERONEU", "AETHER", "AFCONS", "AFFLE", "AFFORDABLE", 
                "AFIL", "AFSL", "AGARIND", "AGARWALEYE", "AGI", 
                "AGIIL", "AGRITECH", "AGROPHOS", "AGSTRA", "AHCL", 
                "AHLADA", "AHLEAST", "AHLUCONT", "AIAENG", "AIIL", 
                "AIRAN", "AIROLAM", "AJANTPHARM", "AJAXENGG", "AJMERA", 
                "AJOONI", "AKASH", "AKG", "AKI", "AKSHAR", 
                "AKSHARCHEM", "AKSHOPTFBR", "AKUMS", "AKZOINDIA", "ALANKIT", 
                "ALBERTDAVD", "ALEMBICLTD", "ALICON", "ALIVUS", "ALKALI", 
                "ALKEM", "ALKYLAMINE", "ALLCARGO", "ALLDIGI", "ALLTIME", 
                "ALMONDZ", "ALOKINDS", "ALPA", "ALPHAGEO", "ALPSINDUS", 
                "AMANTA", "AMBER", "AMBICAAGAR", "AMBIKCO", "AMBUJACEM", 
                "AMDIND", "AMJLAND", "AMNPLST", "AMRUTANJAN", "ANANDRATHI", 
                "ANANTRAJ", "ANDHRAPAP", "ANDHRSUGAR", "ANGELONE", "ANIKINDS", 
                "ANKITMETAL", "ANMOL", "ANSALAPI", "ANTELOPUS", "ANTGRAPHIC", 
                "ANTHEM", "ANUHPHR", "ANUP", "ANURAS", "APARINDS", 
                "APCL", "APCOTEXIND", "APEX", "APLAPOLLO", "APLLTD", 
                "APOLLO", "APOLLOHOSP", "APOLLOPIPE", "APOLLOTYRE", "APOLSINHOT", 
                "APTECHT", "APTUS", "ARCHIDPLY", "ARCHIES", "ARE&M", 
                "ARENTERP", "ARFIN", "ARIES", "ARIHANTCAP", "ARIHANTSUP", 
                "ARISINFRA", "ARKADE", "ARMANFIN", "AROGRANITE", "ARROWGREEN", 
                "ARSHIYA", "ARSSBL", "ARTEMISMED", "ARTNIRMAN", "ARVEE", 
                "ARVIND", "ARVINDFASN", "ARVSMART", "ASAHIINDIA", "ASAHISONG", 
                "ASAL", "ASALCBR", "ASHAPURMIN", "ASHIANA", "ASHIMASYN", 
                "ASHOKA", "ASHOKAMET", "ASHOKLEY", "ASIANENE", "ASIANHOTNR", 
                "ASIANPAINT", "ASIANTILES", "ASKAUTOLTD", "ASMS", "ASPINWALL", 
                "ASTEC", "ASTERDM", "ASTRAL", "ASTRAMICRO", "ASTRAZEN", 
                "ASTRON", "ATALREAL", "ATAM", "ATGL", "ATHERENERG", 
                "ATL", "ATLANTAA", "ATLANTAELE", "ATLASCYCLE", "ATUL", 
                "ATULAUTO", "AUBANK", "AURIGROW", "AURIONPRO", "AUROPHARMA", 
                "AURUM", "AUSOMENT", "AUTOAXLES", "AUTOIND", "AVADHSUGAR", 
                "AVALON", "AVANTEL", "AVANTIFEED", "AVG", "AVL", 
                "AVONMORE", "AVROIND", "AVTNPL", "AWFIS", "AWHCL", 
                "AWL", "AXISBANK", "AXISCADES", "AXITA", "AYMSYNTEX", 
                "AZAD", "BAFNAPH", "BAGFILMS", "BAIDFIN", "BAJAJ-AUTO", 
                "BAJAJCON", "BAJAJELEC", "BAJAJFINSV", "BAJAJHCARE", "BAJAJHFL", 
                "BAJAJHIND", "BAJAJHLDNG", "BAJAJINDEF", "BAJEL", "BAJFINANCE", 
                "BALAJEE", "BALAJITELE", "BALAMINES", "BALAXI", "BALKRISHNA", 
                "BALKRISIND", "BALMLAWRIE", "BALPHARMA", "BALRAMCHIN", "BALUFORGE", 
                "BANARBEADS", "BANARISUG", "BANCOINDIA", "BANDHANBNK", "BANG", 
                "BANKA", "BANKBARODA", "BANKINDIA", "BANSALWIRE", "BANSWRAS", 
                "BASF", "BASML", "BATAINDIA", "BAYERCROP", "BBL", 
                "BBOX", "BBTC", "BBTCL", "BCG", "BCLIND", 
                "BCONCEPTS", "BDL", "BEARDSELL", "BECTORFOOD", "BEDMUTHA", 
                "BEL", "BELLACASA", "BELRISE", "BEML", "BEPL", 
                "BERGEPAINT", "BESTAGRO", "BFINVEST", "BFUTILITIE", "BGLOBAL", 
                "BGRENERGY", "BHAGCHEM", "BHAGERIA", "BHAGYANGR", "BHANDARI", 
                "BHARATFORG", "BHARATGEAR", "BHARATRAS", "BHARATSE", "BHARATWIRE", 
                "BHARTIARTL", "BHARTIHEXA", "BHEL", "BIGBLOC", "BIKAJI", 
                "BIL", "BILVYAPAR", "BIOCON", "BIOFILCHEM", "BIRLACABLE", 
                "BIRLACORPN", "BIRLAMONEY", "BIRLANU", "BLACKBUCK", "BLAL", 
                "BLBLIMITED", "BLISSGVS", "BLKASHYAP", "BLS", "BLSE", 
                "BLUECHIP", "BLUECOAST", "BLUEDART", "BLUEJET", "BLUESTARCO", 
                "BLUESTONE", "BLUSPRING", "BMWVENTLTD", "BODALCHEM", "BOHRAIND", 
                "BOMDYEING", "BORANA", "BOROLTD", "BORORENEW", "BOROSCI", 
                "BOSCHLTD", "BPCL", "BPL", "BRIGADE", "BRIGHOTEL", 
                "BRITANNIA", "BRNL", "BROOKS", "BSE", "BSHSL", 
                "BSL", "BSOFT", "BTML", "BUTTERFLY", "BVCL", 
                "BYKE", "CALSOFT", "CAMLINFINE", "CAMPUS", "CAMS", 
                "CANBK", "CANFINHOME", "CANHLIFE", "CANTABIL", "CAPACITE", 
                "CAPITALSFB", "CAPLIPOINT", "CAPTRUST", "CARBORUNIV", "CARERATING", 
                "CARRARO", "CARTRADE", "CARYSIL", "CASTROLIND", "CCCL", 
                "CCHHL", "CCL", "CDSL", "CEATLTD", "CEIGALL", 
                "CELEBRITY", "CELLO", "CEMPRO", "CENTENKA", "CENTEXT", 
                "CENTRALBK", "CENTRUM", "CENTUM", "CENTURYPLY", "CERA", 
                "CEREBRAINT", "CESC", "CEWATER", "CGCL", "CGPOWER", 
                "CHALET", "CHAMBLFERT", "CHEMBOND", "CHEMBONDCH", "CHEMCON", 
                "CHEMFAB", "CHEMPLASTS", "CHENNPETRO", "CHEVIOT", "CHOICEIN", 
                "CHOLAFIN", "CHOLAHLDNG", "CIEINDIA", "CIFL", "CIGNITITEC", 
                "CINELINE", "CINEVISTA", "CIPLA", "CLEAN", "CLEDUCATE", 
                "CLSEL", "CMICABLES", "CMSINFO", "COALINDIA", "COASTCORP", 
                "COCHINSHIP", "COFFEEDAY", "COFORGE", "COHANCE", "COLPAL", 
                "COMPINFO", "COMPUSOFT", "COMSYN", "CONCOR", "CONCORDBIO", 
                "CONFIPET", "CONSOFINVT", "CONTROLPR", "CORALFINAC", "CORDSCABLE", 
                "COROMANDEL", "COSMOFIRST", "COUNCODOS", "CPCAP", "CPEDU", 
                "CPPLUS", "CRAFTSMAN", "CRAMC", "CREATIVE", "CREATIVEYE", 
                "CREDITACC", "CREST", "CRISIL", "CRIZAC", "CROMPTON", 
                "CROWN", "CSBBANK", "CSLFINANCE", "CTE", "CUB", 
                "CUBEXTUB", "CUMMINSIND", "CUPID", "CURAA", "CYBERMEDIA", 
                "CYBERTECH", "CYIENT", "CYIENTDLM", "DABUR", "DALBHARAT", 
                "DALMIASUG", "DAMCAPITAL", "DAMODARIND", "DANGEE", "DATAMATICS", 
                "DATAPATTNS", "DAVANGERE", "DBCORP", "DBEIL", "DBL", 
                "DBOL", "DBREALTY", "DBSTOCKBRO", "DCAL", "DCBBANK", 
                "DCI", "DCM", "DCMFINSERV", "DCMNVL", "DCMSHRIRAM", 
                "DCMSRIND", "DCW", "DCXINDIA", "DDEVPLSTIK", "DECCANCE", 
                "DEEDEV", "DEEPAKFERT", "DEEPAKNTR", "DEEPINDS", "DELHIVERY", 
                "DELPHIFX", "DELTACORP", "DELTAMAGNT", "DEN", "DENORA", 
                "DENTA", "DEVIT", "DEVX", "DEVYANI", "DGCONTENT", 
                "DHAMPURSUG", "DHANBANK", "DHANUKA", "DHARAN", "DHARMAJ", 
                "DHRUV", "DHUNINV", "DIACABS", "DIAMINESQ", "DIAMONDYD", 
                "DICIND", "DIFFNKG", "DIGIDRIVE", "DIGISPICE", "DIGITIDE", 
                "DIGJAMLMTD", "DIL", "DISHTV", "DIVGIITTS", "DIVISLAB", 
                "DIXON", "DJML", "DLF", "DLINKINDIA", "DMART", 
                "DMCC", "DNAMEDIA", "DODLA", "DOLATALGO", "DOLLAR", 
                "DOLPHIN", "DOMS", "DONEAR", "DPABHUSHAN", "DPSCLTD", 
                "DPWIRES", "DRCSYSTEMS", "DREAMFOLKS", "DREDGECORP", "DRREDDY", 
                "DSSL", "DTIL", "DUCON", "DVL", "DWARKESH", 
                "DYCL", "DYNAMATECH", "DYNPRO", "E2E", "EASEMYTRIP", 
                "EASTSILK", "EBGNG", "ECLERX", "ECOSMOBLTY", "EDELWEISS", 
                "EDUCOMP", "EFCIL", "EICHERMOT", "EIDPARRY", "EIEL", 
                "EIFFL", "EIHAHOTELS", "EIHOTEL", "EIMCOELECO", "EKC", 
                "ELDEHSG", "ELECON", "ELECTCAST", "ELECTHERM", "ELGIEQUIP", 
                "ELGIRUBCO", "ELIN", "ELLEN", "EMAMILTD", "EMAMIPAP", 
                "EMAMIREAL", "EMBDL", "EMCURE", "EMIL", "EMKAY", 
                "EMMBI", "EMSLIMITED", "EMUDHRA", "ENDURANCE", "ENERGYDEV", 
                "ENGINERSIN", "ENIL", "ENRIN", "ENTERO", "EPACK", 
                "EPACKPEB", "EPIGRAL", "EPL", "EQUIPPP", "EQUITASBNK", 
                "ERIS", "ESABINDIA", "ESAFSFB", "ESCORTS", "ESSARSHPNG", 
                "ESSENTIA", "ESTER", "ETERNAL", "ETHOSLTD", "EUREKAFORB", 
                "EUROBOND", "EUROPRATIK", "EUROTEXIND", "EVEREADY", "EVERESTIND", 
                "EXCEL", "EXCELINDUS", "EXICOM", "EXIDEIND", "EXPLEOSOL", 
                "EXXARO", "FABTECH", "FACT", "FAIRCHEMOR", "FAZE3Q", 
                "FCL", "FCONSUMER", "FCSSOFT", "FDC", "FEDERALBNK", 
                "FEDFINA", "FEL", "FELDVR", "FIBERWEB", "FIEMIND", 
                "FILATEX", "FILATFASH", "FINCABLES", "FINEORG", "FINKURVE", 
                "FINOPB", "FINPIPE", "FIRSTCRY", "FISCHER", "FIVESTAR", 
                "FLAIR", "FLEXITUFF", "FLFL", "FLUOROCHEM", "FMGOETZE", 
                "FMNL", "FOCUS", "FOODSIN", "FORCEMOT", "FORTIS", 
                "FOSECOIND", "FSL", "FUSION", "GABRIEL", "GAEL", 
                "GAIL", "GALAPREC", "GALAXYSURF", "GALLANTT", "GANDHAR", 
                "GANDHITUBE", "GANECOS", "GANESHBE", "GANESHCP", "GANESHHOU", 
                "GANGAFORGE", "GANGESSECU", "GARFIBRES", "GARUDA", "GATECH", 
                "GATECHDVR", "GATEWAY", "GAYAHWS", "GAYAPROJ", "GCSL", 
                "GEECEE", "GEEKAYWIRE", "GEMAROMA", "GENCON", "GENESYS", 
                "GENSOL", "GENUSPAPER", "GENUSPOWER", "GEOJITFSL", "GESHIP", 
                "GFLLIMITED", "GFSTEELS", "GHCL", "GHCLTEXTIL", "GICHSGFIN", 
                "GICRE", "GILLANDERS", "GILLETTE", "GINNIFILA", "GIPCL", 
                "GKENERGY", "GKWLIMITED", "GLAND", "GLAXO", "GLENMARK", 
                "GLFL", "GLOBAL", "GLOBALE", "GLOBALVECT", "GLOBE", 
                "GLOBECIVIL", "GLOBUSSPR", "GLOSTERLTD", "GLOTTIS", "GMBREW", 
                "GMDCLTD", "GMMPFAUDLR", "GMRAIRPORT", "GMRP&UI", "GNA", 
                "GNFC", "GOACARBON", "GOCLCORP", "GOCOLORS", "GODAVARIB", 
                "GODFRYPHLP", "GODIGIT", "GODREJAGRO", "GODREJCP", "GODREJIND", 
                "GODREJPROP", "GOENKA", "GOKEX", "GOKUL", "GOKULAGRO", 
                "GOLDENTOBC", "GOLDIAM", "GOLDTECH", "GOODLUCK", "GOPAL", 
                "GOYALALUM", "GPIL", "GPPL", "GPTHEALTH", "GPTINFRA", 
                "GRANULES", "GRAPHITE", "GRASIM", "GRAVITA", "GREAVESCOT", 
                "GREENLAM", "GREENPANEL", "GREENPLY", "GREENPOWER", "GRINDWELL", 
                "GRINFRA", "GRMOVER", "GROBTEA", "GROWW", "GRPLTD", 
                "GRSE", "GRWRHITECH", "GSFC", "GSLSU", "GSPL", 
                "GSS", "GTECJAINX", "GTL", "GTLINFRA", "GTPL", 
                "GUFICBIO", "GUJALKALI", "GUJAPOLLO", "GUJGASLTD", "GUJRAFFIA", 
                "GUJTHEM", "GULFOILLUB", "GULFPETRO", "GULPOLY", "GVKPIL", 
                "GVPIL", "GVPTECH", "GVT&D", "HAL", "HAPPSTMNDS", 
                "HAPPYFORGE", "HARDWYN", "HARIOMPIPE", "HARRMALAYA", "HARSHA", 
                "HATHWAY", "HATSUN", "HAVELLS", "HAVISHA", "HBLENGINE", 
                "HBSL", "HCC", "HCG", "HCL-INSYS", "HCLTECH", 
                "HDBFS", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HDIL", 
                "HEADSUP", "HECPROJECT", "HEG", "HEIDELBERG", "HEMIPROP", 
                "HERANBA", "HERCULES", "HERITGFOOD", "HEROMOTOCO", "HESTERBIO", 
                "HEUBACHIND", "HEXATRADEX", "HEXT", "HFCL", "HGINFRA", 
                "HGM", "HGS", "HIKAL", "HILINFRA", "HILTON", 
                "HIMATSEIDE", "HINDALCO", "HINDCOMPOS", "HINDCON", "HINDCOPPER", 
                "HINDOILEXP", "HINDPETRO", "HINDUNILVR", "HINDWAREAP", "HINDZINC", 
                "HIRECT", "HISARMETAL", "HITECH", "HITECHCORP", "HITECHGEAR", 
                "HLEGLAS", "HLVLTD", "HMAAGRO", "HMT", "HMVL", 
                "HNDFDS", "HOMEFIRST", "HONASA", "HONAUT", "HONDAPOWER", 
                "HPAL", "HPIL", "HPL", "HSCL", "HTMEDIA", 
                "HUBTOWN", "HUDCO", "HUHTAMAKI", "HYBRIDFIN", "HYUNDAI", 
                "IBULLSLTD", "ICDSLTD", "ICEMAKE", "ICICIBANK", "ICICIGI", 
                "ICICIPRULI", "ICIL", "ICRA", "IDBI", "IDEA", 
                "IDEAFORGE", "IDFCFIRSTB", "IEX", "IFBAGRO", "IFBIND", 
                "IFCI", "IFGLEXPOR", "IGARASHI", "IGCL", "IGIL", 
                "IGL", "IGPL", "IIFL", "IIFLCAPS", "IITL", 
                "IKIO", "IKS", "IL&FSENGG", "IL&FSTRANS", "IMAGICAA", 
                "IMFA", "IMPAL", "IMPEXFERRO", "INCREDIBLE", "INDBANK", 
                "INDGN", "INDHOTEL", "INDIACEM", "INDIAGLYCO", "INDIAMART", 
                "INDIANB", "INDIANCARD", "INDIANHUME", "INDIASHLTR", "INDIGO", 
                "INDIGOPNTS", "INDIQUBE", "INDNIPPON", "INDOAMIN", "INDOBORAX", 
                "INDOCO", "INDOFARM", "INDORAMA", "INDOSTAR", "INDOTECH", 
                "INDOTHAI", "INDOUS", "INDOWIND", "INDRAMEDCO", "INDSWFTLAB", 
                "INDTERRAIN", "INDUSINDBK", "INDUSTOWER", "INFIBEAM", "INFOBEAN", 
                "INFOMEDIA", "INFY", "INGERRAND", "INNOVACAP", "INNOVANA", 
                "INOXGREEN", "INOXINDIA", "INOXWIND", "INSECTICID", "INSPIRISYS", 
                "INTELLECT", "INTENTECH", "INTERARCH", "INTLCONV", "INVENTURE", 
                "IOB", "IOC", "IOLCP", "IONEXCHANG", "IPCALAB", 
                "IPL", "IRB", "IRCON", "IRCTC", "IREDA", 
                "IRFC", "IRIS", "IRISDOREME", "IRMENERGY", "ISFT", 
                "ISGEC", "ISHANCH", "ITC", "ITCHOTELS", "ITDC", 
                "ITI", "IVALUE", "IVC", "IVP", "IXIGO", 
                "IZMO", "J&KBANK", "JAGRAN", "JAGSNPHARM", "JAIBALAJI", 
                "JAICORPLTD", "JAINREC", "JAIPURKURT", "JAMNAAUTO", "JARO", 
                "JASH", "JAYAGROGN", "JAYBARMARU", "JAYNECOIND", "JAYSREETEA", 
                "JBCHEPHARM", "JBMA", "JCHAC", "JETFREIGHT", "JGCHEM", 
                "JHS", "JINDALPHOT", "JINDALPOLY", "JINDALSAW", "JINDALSTEL", 
                "JINDRILL", "JINDWORLD", "JIOFIN", "JISLDVREQS", "JISLJALEQS", 
                "JITFINFRA", "JKCEMENT", "JKIL", "JKIPL", "JKLAKSHMI", 
                "JKPAPER", "JKTYRE", "JLHL", "JMA", "JMFINANCIL", 
                "JNKINDIA", "JOCIL", "JPASSOCIAT", "JPOLYINVST", "JPPOWER", 
                "JSFB", "JSL", "JSLL", "JSWCEMENT", "JSWENERGY", 
                "JSWHL", "JSWINFRA", "JSWSTEEL", "JTEKTINDIA", "JTLIND", 
                "JUBLCPL", "JUBLFOOD", "JUBLINGREA", "JUBLPHARMA", "JUNIPER", 
                "JUSTDIAL", "JWL", "JYOTHYLAB", "JYOTICNC", "JYOTISTRUC", 
                "KABRAEXTRU", "KAJARIACER", "KAKATCEM", "KALAMANDIR", "KALPATARU", 
                "KALYANI", "KALYANIFRG", "KALYANKJIL", "KAMATHOTEL", "KAMDHENU", 
                "KAMOPAINTS", "KANANIIND", "KANORICHEM", "KANPRPLA", "KANSAINER", 
                "KAPSTON", "KARMAENG", "KARURVYSYA", "KAUSHALYA", "KAVDEFENCE", 
                "KAYA", "KAYNES", "KCP", "KCPSUGIND", "KDDL", 
                "KEC", "KECL", "KEEPLEARN", "KEI", "KELLTONTEC", 
                "KERNEX", "KESORAMIND", "KEYFINSERV", "KFINTECH", "KHADIM", 
                "KHAICHEM", "KHAITANLTD", "KHANDSE", "KICL", "KILITCH", 
                "KIMS", "KINGFA", "KIOCL", "KIRIINDUS", "KIRLOSBROS", 
                "KIRLOSENG", "KIRLOSIND", "KIRLPNU", "KITEX", "KKCL", 
                "KMEW", "KMSUGAR", "KNRCON", "KOHINOOR", "KOKUYOCMLN", 
                "KOLTEPATIL", "KOPRAN", "KOTAKBANK", "KOTARISUG", "KOTHARIPET", 
                "KOTHARIPRO", "KPEL", "KPIGREEN", "KPIL", "KPITTECH", 
                "KPRMILL", "KRBL", "KREBSBIO", "KRIDHANINF", "KRISHANA", 
                "KRISHIVAL", "KRITI", "KRITIKA", "KRITINUT", "KRN", 
                "KRONOX", "KROSS", "KRSNAA", "KRYSTAL", "KSB", 
                "KSCL", "KSHITIJPOL", "KSL", "KSOLVES", "KTKBANK", 
                "KUANTUM", "LAGNAM", "LAKPRE", "LAL", "LALPATHLAB", 
                "LAMBODHARA", "LANCORHOL", "LANDMARK", "LAOPALA", "LASA", 
                "LATENTVIEW", "LATTEYS", "LAURUSLABS", "LAXMICOT", "LAXMIDENTL", 
                "LAXMIINDIA", "LCCINFOTEC", "LEMONTREE", "LENSKART", "LEXUS", 
                "LFIC", "LGBBROSLTD", "LGEINDIA", "LGHL", "LIBAS", 
                "LIBERTSHOE", "LICHSGFIN", "LICI", "LIKHITHA", "LINC", 
                "LINCOLN", "LINDEINDIA", "LLOYDSENGG", "LLOYDSENT", "LLOYDSME", 
                "LMW", "LODHA", "LOKESHMACH", "LORDSCHLO", "LOTUSDEV", 
                "LOTUSEYE", "LOVABLE", "LOYALTEX", "LPDC", "LT", 
                "LTF", "LTFOODS", "LTIM", "LTTS", "LUMAXIND", 
                "LUMAXTECH", "LUPIN", "LUXIND", "LXCHEM", "LYKALABS", 
                "LYPSAGEMS", "M&M", "M&MFIN", "MAANALU", "MACPOWER", 
                "MADHAV", "MADHUCON", "MADRASFERT", "MAGADSUGAR", "MAGNUM", 
                "MAHABANK", "MAHAPEXLTD", "MAHASTEEL", "MAHEPC", "MAHESHWARI", 
                "MAHLIFE", "MAHLOG", "MAHSCOOTER", "MAHSEAMLES", "MAITHANALL", 
                "MALLCOM", "MALUPAPER", "MAMATA", "MANAKALUCO", "MANAKCOAT", 
                "MANAKSIA", "MANAKSTEEL", "MANALIPETC", "MANAPPURAM", "MANBA", 
                "MANCREDIT", "MANGALAM", "MANGLMCEM", "MANINDS", "MANINFRA", 
                "MANKIND", "MANOMAY", "MANORAMA", "MANORG", "MANUGRAPH", 
                "MANYAVAR", "MAPMYINDIA", "MARALOVER", "MARATHON", "MARICO", 
                "MARINE", "MARKOLINES", "MARKSANS", "MARSHALL", "MARUTI", 
                "MASFIN", "MASKINVEST", "MASTEK", "MASTERTR", "MATRIMONY", 
                "MAWANASUG", "MAXESTATES", "MAXHEALTH", "MAXIND", "MAYURUNIQ", 
                "MAZDA", "MAZDOCK", "MBAPL", "MBEL", "MBLINFRA", 
                "MCL", "MCLEODRUSS", "MCLOUD", "MCX", "MEDANTA", 
                "MEDIASSIST", "MEDICAMEQ", "MEDICO", "MEDPLUS", "MEGASOFT", 
                "MEGASTAR", "MEIL", "MENONBE", "MEP", "METROBRAND", 
                "METROPOLIS", "MFML", "MFSL", "MGEL", "MGL", 
                "MHLXMIRU", "MHRIL", "MICEL", "MIDHANI", "MIDWESTLTD", 
                "MINDACORP", "MINDTECK", "MIRCELECTR", "MIRZAINT", "MITCON", 
                "MITTAL", "MKPL", "MMFL", "MMP", "MMTC", 
                "MOBIKWIK", "MODIRUBBER", "MODIS", "MODISONLTD", "MODTHREAD", 
                "MOHITIND", "MOIL", "MOKSH", "MOL", "MOLDTECH", 
                "MOLDTKPAC", "MONARCH", "MONEYBOXX", "MONTECARLO", "MORARJEE", 
                "MOREPENLAB", "MOSCHIP", "MOTHERSON", "MOTILALOFS", "MOTISONS", 
                "MOTOGENFIN", "MPHASIS", "MPSLTD", "MRF", "MRPL", 
                "MSPL", "MSTCLTD", "MSUMI", "MTARTECH", "MTEDUCARE", 
                "MTNL", "MUFIN", "MUFTI", "MUKANDLTD", "MUKKA", 
                "MUKTAARTS", "MUNJALAU", "MUNJALSHOW", "MURUDCERA", "MUTHOOTCAP", 
                "MUTHOOTFIN", "MUTHOOTMF", "MVGJL", "MWL", "NACLIND", 
                "NAGAFERT", "NAGREEKCAP", "NAGREEKEXP", "NAHARCAP", "NAHARINDUS", 
                "NAHARPOLY", "NAHARSPING", "NAM-INDIA", "NARMADA", "NATCAPSUQ", 
                "NATCOPHARM", "NATHBIOGEN", "NATIONALUM", "NAUKRI", "NAVA", 
                "NAVINFLUOR", "NAVKARCORP", "NAVKARURB", "NAVNETEDUL", "NAZARA", 
                "NBCC", "NBIFIN", "NCC", "NCLIND", "NDGL", 
                "NDL", "NDLVENTURE", "NDRAUTO", "NDTV", "NECCLTD", 
                "NECLIFE", "NELCAST", "NELCO", "NEOGEN", "NESCO", 
                "NESTLEIND", "NETWEB", "NETWORK18", "NEULANDLAB", "NEWGEN", 
                "NEXTMEDIA", "NFL", "NGIL", "NGLFINE", "NH", 
                "NHPC", "NIACL", "NIBE", "NIBL", "NIITLTD", 
                "NIITMTS", "NILAINFRA", "NILASPACES", "NILKAMAL", "NINSYS", 
                "NIPPOBATRY", "NIRAJ", "NIRAJISPAT", "NITCO", "NITINSPIN", 
                "NITIRAJ", "NIVABUPA", "NKIND", "NLCINDIA", "NMDC", 
                "NOCIL", "NOIDATOLL", "NORBTEAEXP", "NORTHARC", "NOVAAGRI", 
                "NPST", "NRAIL", "NRBBEARING", "NRL", "NSIL", 
                "NSLNISP", "NTPC", "NTPCGREEN", "NUCLEUS", "NURECA", 
                "NUVAMA", "NUVOCO", "NYKAA", "OAL", "OBCL", 
                "OBEROIRLTY", "OCCLLTD", "ODIGMA", "OFSS", "OIL", 
                "OILCOUNTUB", "OLAELEC", "OLECTRA", "OMAXAUTO", "OMAXE", 
                "OMFREIGHT", "OMINFRAL", "OMKARCHEM", "ONELIFECAP", "ONEPOINT", 
                "ONESOURCE", "ONGC", "ONMOBILE", "ONWARDTEC", "OPTIEMUS", 
                "ORBTEXP", "ORCHASP", "ORCHPHARMA", "ORICONENT", "ORIENTALTL", 
                "ORIENTBELL", "ORIENTCEM", "ORIENTCER", "ORIENTELEC", "ORIENTHOT", 
                "ORIENTLTD", "ORIENTPPR", "ORIENTTECH", "ORISSAMINE", "ORKLAINDIA", 
                "ORTEL", "ORTINGLOBE", "OSIAHYPER", "OSWALAGRO", "OSWALGREEN", 
                "OSWALPUMPS", "OSWALSEEDS", "PACEDIGITK", "PAGEIND", "PAISALO", 
                "PAKKA", "PALASHSECU", "PALREDTEC", "PANACEABIO", "PANACHE", 
                "PANAMAPET", "PANSARI", "PAR", "PARACABLES", "PARADEEP", 
                "PARAGMILK", "PARAS", "PARASPETRO", "PARKHOTELS", "PARSVNATH", 
                "PASHUPATI", "PASUPTAC", "PATANJALI", "PATELENG", "PATELRMART", 
                "PATINTLOG", "PAVNAIND", "PAYTM", "PCBL", "PCJEWELLER", 
                "PDMJEPAPER", "PDSL", "PEARLPOLY", "PENIND", "PENINLAND", 
                "PERSISTENT", "PETRONET", "PFC", "PFIZER", "PFOCUS", 
                "PFS", "PGEL", "PGHH", "PGHL", "PGIL", 
                "PHOENIXLTD", "PICCADIL", "PIDILITIND", "PIGL", "PIIND", 
                "PILANIINVS", "PILITA", "PINELABS", "PIONEEREMB", "PIRAMALFIN", 
                "PITTIENG", "PIXTRANS", "PKTEA", "PLASTIBLEN", "PLATIND", 
                "PLAZACABLE", "PNB", "PNBGILTS", "PNBHOUSING", "PNC", 
                "PNCINFRA", "PNGJL", "POCL", "PODDARMENT", "POKARNA", 
                "POLICYBZR", "POLYCAB", "POLYMED", "POLYPLEX", "PONNIERODE", 
                "POONAWALLA", "POWERGRID", "POWERINDIA", "POWERMECH", "PPAP", 
                "PPL", "PPLPHARMA", "PRABHA", "PRAENG", "PRAJIND", 
                "PRAKASH", "PRAKASHSTL", "PRAXIS", "PRECAM", "PRECOT", 
                "PRECWIRE", "PREMEXPLN", "PREMIER", "PREMIERENE", "PREMIERPOL", 
                "PRESTIGE", "PRICOLLTD", "PRIMESECU", "PRIMO", "PRINCEPIPE", 
                "PRITI", "PRITIKAUTO", "PRIVISCL", "PROSTARM", "PROTEAN", 
                "PROZONER", "PRSMJOHNSN", "PRUDENT", "PRUDMOULI", "PSB", 
                "PSPPROJECT", "PTC", "PTCIL", "PTL", "PUNJABCHEM", 
                "PURVA", "PVP", "PVRINOX", "PVSL", "PYRAMID", 
                "QPOWER", "QUADFUTURE", "QUESS", "QUICKHEAL", "QUINTEGRA", 
                "RACE", "RACLGEAR", "RADAAN", "RADHIKAJWE", "RADIANTCMS", 
                "RADICO", "RADIOCITY", "RAILTEL", "RAIN", "RAINBOW", 
                "RAJESHEXPO", "RAJMET", "RAJOOENG", "RAJRATAN", "RAJRILTD", 
                "RAJSREESUG", "RAJTV", "RAJVIR", "RALLIS", "RAMANEWS", 
                "RAMAPHO", "RAMASTEEL", "RAMCOCEM", "RAMCOIND", "RAMCOSYS", 
                "RAMKY", "RAMRAT", "RANASUG", "RANEHOLDIN", "RATEGAIN", 
                "RATNAMANI", "RATNAVEER", "RAYMOND", "RAYMONDLSL", "RAYMONDREL", 
                "RBA", "RBLBANK", "RBZJEWEL", "RCF", "RCOM", 
                "RECLTD", "REDINGTON", "REDTAPE", "REFEX", "REGAAL", 
                "REGENCERAM", "RELAXO", "RELCHEMQ", "RELIABLE", "RELIANCE", 
                "RELIGARE", "RELINFRA", "RELTD", "REMSONSIND", "RENUKA", 
                "REPCOHOME", "REPL", "REPRO", "RESPONIND", "RETAIL", 
                "RGL", "RHETAN", "RHFL", "RHIM", "RHL", 
                "RICOAUTO", "RIIL", "RISHABH", "RITCO", "RITES", 
                "RKDL", "RKEC", "RKFORGE", "RKSWAMY", "RMDRIP", 
                "RML", "RNBDENIMS", "ROHLTD", "ROLEXRINGS", "ROLLT", 
                "ROLTA", "ROML", "ROSSARI", "ROSSELLIND", "ROSSTECH", 
                "ROTO", "ROUTE", "RPEL", "RPGLIFE", "RPOWER", 
                "RPPINFRA", "RPPL", "RPSGVENT", "RPTECH", "RRKABEL", 
                "RSSOFTWARE", "RSWM", "RSYSTEMS", "RTNINDIA", "RTNPOWER", 
                "RUBFILA", "RUBICON", "RUBYMILLS", "RUCHINFRA", "RUCHIRA", 
                "RUPA", "RUSHIL", "RUSTOMJEE", "RVHL", "RVNL", 
                "RVTH", "S&SPOWER", "SAATVIKGL", "SABEVENTS", "SABTNL", 
                "SADBHAV", "SADBHIN", "SADHNANIQ", "SAFARI", "SAGARDEEP", 
                "SAGCEM", "SAGILITY", "SAHYADRI", "SAIL", "SAILIFE", 
                "SAKAR", "SAKHTISUG", "SAKSOFT", "SAKUMA", "SALASAR", 
                "SALONA", "SALSTEEL", "SALZERELEC", "SAMBHAAV", "SAMBHV", 
                "SAMHI", "SAMMAANCAP", "SAMPANN", "SANATHAN", "SANCO", 
                "SANDESH", "SANDHAR", "SANDUMA", "SANGAMIND", "SANGHIIND", 
                "SANGHVIMOV", "SANGINITA", "SANOFI", "SANOFICONR", "SANSERA", 
                "SANSTAR", "SANWARIA", "SAPPHIRE", "SARDAEN", "SAREGAMA", 
                "SARLAPOLY", "SARVESHWAR", "SASKEN", "SASTASUNDR", "SATIA", 
                "SATIN", "SAURASHCEM", "SBC", "SBCL", "SBFC", 
                "SBGLP", "SBICARD", "SBILIFE", "SBIN", "SCHAEFFLER", 
                "SCHAND", "SCHNEIDER", "SCI", "SCILAL", "SCODATUBES", 
                "SCPL", "SDBL", "SEAMECLTD", "SECMARK", "SECURKLOUD", 
                "SEJALLTD", "SELMC", "SEMAC", "SENCO", "SENORES", 
                "SEPC", "SEQUENT", "SERVOTECH", "SESHAPAPER", "SETCO", 
                "SETUINFRA", "SEYAIND", "SFL", "SGFIN", "SGIL", 
                "SGL", "SGLTL", "SGMART", "SHAH", "SHAHALLOYS", 
                "SHAILY", "SHAKTIPUMP", "SHALBY", "SHALPAINTS", "SHANKARA", 
                "SHANTI", "SHANTIGEAR", "SHANTIGOLD", "SHARDACROP", "SHARDAMOTR", 
                "SHAREINDIA", "SHEKHAWATI", "SHEMAROO", "SHILPAMED", "SHIVALIK", 
                "SHIVAMAUTO", "SHIVAMILLS", "SHIVATEX", "SHIVAUM", "SHK", 
                "SHOPERSTOP", "SHRADHA", "SHREDIGCEM", "SHREECEM", "SHREEJISPG", 
                "SHREEPUSHK", "SHREERAMA", "SHRENIK", "SHREYANIND", "SHRINGARMS", 
                "SHRIPISTON", "SHRIRAMFIN", "SHRIRAMPPS", "SHYAMCENT", "SHYAMMETL", 
                "SHYAMTEL", "SICALLOG", "SIEMENS", "SIGACHI", "SIGIND", 
                "SIGMA", "SIGNATURE", "SIGNPOST", "SIKKO", "SIL", 
                "SILGO", "SILINV", "SILLYMONKS", "SILVERTUC", "SIMBHALS", 
                "SIMPLEXINF", "SINCLAIR", "SINDHUTRAD", "SINTERCOM", "SIRCA", 
                "SIS", "SITINET", "SIYSIL", "SJS", "SJVN", 
                "SKFINDIA", "SKIL", "SKIPPER", "SKMEGGPROD", "SKYGOLD", 
                "SMARTLINK", "SMARTWORKS", "SMCGLOBAL", "SMLMAH", "SMLT", 
                "SMSLIFE", "SMSPHARMA", "SNOWMAN", "SOBHA", "SOFTTECH", 
                "SOLARA", "SOLARINDS", "SOLARWORLD", "SOLEX", "SOMANYCERA", 
                "SOMATEX", "SOMICONVEY", "SONACOMS", "SONAMLTD", "SONATSOFTW", 
                "SOTL", "SOUTHBANK", "SOUTHWEST", "SPAL", "SPANDANA", 
                "SPARC", "SPCENET", "SPECIALITY", "SPECTRUM", "SPENCERS", 
                "SPIC", "SPLIL", "SPLPETRO", "SPMLINFRA", "SPORTKING", 
                "SRD", "SREEL", "SRF", "SRGHFL", "SRHHYPOLTD", 
                "SRM", "SRPL", "SSDL", "SSWL", "STALLION", 
                "STANLEY", "STAR", "STARCEMENT", "STARHEALTH", "STARPAPER", 
                "STARTECK", "STCINDIA", "STEELCAS", "STEELCITY", "STEELXIND", 
                "STEL", "STERTOOLS", "STLNETWORK", "STLTECH", "STOVEKRAFT", 
                "STUDDS", "STYL", "STYLAMIND", "STYLEBAAZA", "STYRENIX", 
                "SUBEXLTD", "SUBROS", "SUDARSCHEM", "SUKHJITS", "SULA", 
                "SUMEETINDS", "SUMICHEM", "SUMIT", "SUMMITSEC", "SUNCLAY", 
                "SUNDARAM", "SUNDARMFIN", "SUNDRMBRAK", "SUNDRMFAST", "SUNDROP", 
                "SUNFLAG", "SUNPHARMA", "SUNTECK", "SUNTV", "SUPERHOUSE", 
                "SUPERSPIN", "SUPRAJIT", "SUPREME", "SUPREMEENG", "SUPREMEIND", 
                "SUPREMEINF", "SUPRIYA", "SURAJEST", "SURAJLTD", "SURAKSHA", 
                "SURANASOL", "SURANAT&P", "SURYALAXMI", "SURYAROSNI", "SURYODAY", 
                "SUTLEJTEX", "SUULD", "SUVEN", "SUVIDHAA", "SUYOG", 
                "SUZLON", "SVLL", "SVPGLOB", "SWANCORP", "SWANDEF", 
                "SWARAJENG", "SWELECTES", "SWIGGY", "SWSOLAR", "SYMPHONY", 
                "SYNCOMF", "SYNGENE", "SYRMA", "SYSTMTXC", "TAINWALCHM", 
                "TAJGVK", "TAKE", "TALBROAUTO", "TANLA", "TARACHAND", 
                "TARAPUR", "TARC", "TARIL", "TARMAT", "TARSONS", 
                "TASTYBITE", "TATACAP", "TATACHEM", "TATACOMM", "TATACONSUM", 
                "TATAELXSI", "TATAINVEST", "TATAPOWER", "TATASTEEL", "TATATECH", 
                "TATVA", "TBOTEK", "TBZ", "TCI", "TCIEXP", 
                "TCIFINANCE", "TCPLPACK", "TCS", "TDPOWERSYS", "TEAMGTY", 
                "TEAMLEASE", "TECHM", "TECHNOE", "TECILCHEM", "TEGA", 
                "TEJASNET", "TEMBO", "TERASOFT", "TEXINFRA", "TEXMOPIPES", 
                "TEXRAIL", "TFCILTD", "TFL", "TGBHOTELS", "THANGAMAYL", 
                "THEINVEST", "THEJO", "THELEELA", "THEMISMED", "THERMAX", 
                "THOMASCOOK", "THOMASCOTT", "THYROCARE", "TI", "TICL", 
                "TIGERLOGS", "TIIL", "TIINDIA", "TIJARIA", "TIL", 
                "TIMETECHNO", "TIMKEN", "TINNARUBR", "TIPSFILMS", "TIPSMUSIC", 
                "TIRUMALCHM", "TIRUPATIFL", "TITAGARH", "TITAN", "TMB", 
                "TMCV", "TMPV", "TNPETRO", "TNPL", "TNTELE", 
                "TOKYOPLAST", "TOLINS", "TORNTPHARM", "TORNTPOWER", "TOTAL", 
                "TOUCHWOOD", "TPHQ", "TPLPLASTEH", "TRACXN", "TRANSRAILL", 
                "TRANSWORLD", "TRAVELFOOD", "TREEHOUSE", "TREJHARA", "TREL", 
                "TRENT", "TRF", "TRIDENT", "TRIGYN", "TRITURBINE", 
                "TRIVENI", "TRU", "TRUALT", "TSFINV", "TTKHLTCARE", 
                "TTKPRESTIG", "TTL", "TTML", "TVSELECT", "TVSHLTD", 
                "TVSMOTOR", "TVSSCS", "TVSSRICHAK", "TVTODAY", "TVVISION", 
                "UBL", "UCAL", "UCOBANK", "UDS", "UEL", 
                "UFBL", "UFLEX", "UFO", "UGARSUGAR", "UGROCAP", 
                "UJJIVANSFB", "ULTRACEMCO", "UMAEXPORTS", "UMESLTD", "UMIYA-MRO", 
                "UNICHEMLAB", "UNIDT", "UNIECOM", "UNIENTER", "UNIINFO", 
                "UNIMECH", "UNIONBANK", "UNIPARTS", "UNITDSPR", "UNITECH", 
                "UNITEDPOLY", "UNITEDTEA", "UNIVAFOODS", "UNIVASTU", "UNIVCABLES", 
                "UNIVPHOTO", "UNOMINDA", "UPL", "URAVIDEF", "URBANCO", 
                "URJA", "USHAMART", "USK", "UTIAMC", "UTKARSHBNK", 
                "UTTAMSUGAR", "UYFINCORP", "V2RETAIL", "VADILALIND", "VAIBHAVGBL", 
                "VAISHALI", "VAKRANGEE", "VALIANTLAB", "VALIANTORG", "VARDHACRLC", 
                "VARDMNPOLY", "VARROC", "VASCONEQ", "VASWANI", "VBL", 
                "VCL", "VEDL", "VEEDOL", "VENKEYS", "VENTIVE", 
                "VENUSPIPES", "VENUSREM", "VERANDA", "VERTOZ", "VESUVIUS", 
                "VETO", "VGL", "VGUARD", "VHL", "VHLTD", 
                "VIDHIING", "VIJAYA", "VIJIFIN", "VIKASECO", "VIKASLIFE", 
                "VIKRAMSOLR", "VIKRAN", "VIMTALABS", "VINATIORGA", "VINCOFE", 
                "VINDHYATEL", "VINEETLAB", "VINNY", "VINYLINDIA", "VIPCLOTHNG", 
                "VIPIND", "VIPULLTD", "VIRINCHI", "VISAKAIND", "VISASTEEL", 
                "VISHNU", "VISHWARAJ", "VIVIDHA", "VLEGOV", "VLSFINANCE", 
                "VMART", "VMM", "VMSTMT", "VOLTAMP", "VOLTAS", 
                "VPRPL", "VRAJ", "VRLLOG", "VSSL", "VSTIND", 
                "VSTL", "VSTTILLERS", "VTL", "WAAREEENER", "WAAREEINDO", 
                "WAAREERTL", "WABAG", "WALCHANNAG", "WANBURY", "WCIL", 
                "WEALTH", "WEBELSOLAR", "WEIZMANIND", "WEL", "WELCORP", 
                "WELENT", "WELINV", "WELSPUNLIV", "WENDT", "WESTLIFE", 
                "WEWIN", "WEWORK", "WHEELS", "WHIRLPOOL", "WILLAMAGOR", 
                "WINDLAS", "WINDMACHIN", "WINSOME", "WIPL", "WIPRO", 
                "WOCKPHARMA", "WONDERLA", "WORTHPERI", "WSI", "WSTCSTPAPR", 
                "XCHANGING", "XELPMOC", "XPROINDIA", "XTGLOBAL", "YASHO", 
                "YATHARTH", "YATRA", "YESBANK", "YUKEN", "ZAGGLE", 
                "ZEEL", "ZEELEARN", "ZEEMEDIA", "ZENITHEXPO", "ZENITHSTL", 
                "ZENSARTECH", "ZENTEC", "ZFCVINDIA", "ZIMLAB", "ZODIAC", 
                "ZODIACLOTH", "ZOTA", "ZUARI", "ZUARIIND", "ZYDUSLIFE", 
                "ZYDUSWELL"
]

SECTOR_MAP = {
    'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT', 'ICICIBANK': 'Banking',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'SBIN': 'Banking', 'BHARTIARTL': 'Telecom', 'KOTAKBANK': 'Banking',
    'LT': 'Infrastructure', 'AXISBANK': 'Banking', 'ASIANPAINT': 'Paints', 'MARUTI': 'Auto', 'HCLTECH': 'IT',
    'BAJFINANCE': 'NBFC', 'WIPRO': 'IT', 'SUNPHARMA': 'Pharma', 'TITAN': 'Consumer', 'ULTRACEMCO': 'Cement',
    'NESTLEIND': 'FMCG', 'ONGC': 'Energy', 'TATAMOTORS': 'Auto', 'NTPC': 'Power', 'POWERGRID': 'Power',
    'JSWSTEEL': 'Metals', 'M&M': 'Auto', 'TECHM': 'IT', 'ADANIENT': 'Conglomerate', 'ADANIPORTS': 'Infrastructure'
}

@st.cache_data(ttl=300)
def fetch_stock_data(symbol):
    """Fetch real-time data from Yahoo Finance with fundamentals"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="3mo", interval="1d")
        
        if hist.empty:
            return None
        
        closes = hist['Close'].values
        highs = hist['High'].values
        lows = hist['Low'].values
        volumes = hist['Volume'].values
        
        price = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else price
        change = ((price - prev_close) / prev_close) * 100
        
        # Get fundamental data
        info = ticker.info
        market_cap = info.get('marketCap', 0)
        
        # Revenue and profit growth
        revenue_growth = info.get('revenueGrowth', None)
        profit_margin = info.get('profitMargins', None)
        earnings_growth = info.get('earningsGrowth', None)
        
        # Additional fundamental metrics
        pe_ratio = info.get('trailingPE', None)
        roe = info.get('returnOnEquity', None)
        debt_to_equity = info.get('debtToEquity', None)
        
        # Get quarterly financials for growth analysis
        try:
            financials = ticker.quarterly_financials
            if financials is not None and not financials.empty:
                # Get Total Revenue if available
                if 'Total Revenue' in financials.index:
                    revenues = financials.loc['Total Revenue'].values
                    if len(revenues) >= 4:
                        # Calculate QoQ and YoY growth
                        qoq_revenue_growth = ((revenues[0] - revenues[1]) / abs(revenues[1])) * 100 if revenues[1] != 0 else None
                        yoy_revenue_growth = ((revenues[0] - revenues[3]) / abs(revenues[3])) * 100 if len(revenues) >= 4 and revenues[3] != 0 else None
                    else:
                        qoq_revenue_growth = None
                        yoy_revenue_growth = None
                else:
                    qoq_revenue_growth = None
                    yoy_revenue_growth = None
                
                # Get Net Income if available
                if 'Net Income' in financials.index:
                    profits = financials.loc['Net Income'].values
                    if len(profits) >= 4:
                        qoq_profit_growth = ((profits[0] - profits[1]) / abs(profits[1])) * 100 if profits[1] != 0 else None
                        yoy_profit_growth = ((profits[0] - profits[3]) / abs(profits[3])) * 100 if len(profits) >= 4 and profits[3] != 0 else None
                    else:
                        qoq_profit_growth = None
                        yoy_profit_growth = None
                else:
                    qoq_profit_growth = None
                    yoy_profit_growth = None
            else:
                qoq_revenue_growth = None
                yoy_revenue_growth = None
                qoq_profit_growth = None
                yoy_profit_growth = None
        except Exception as e:
            qoq_revenue_growth = None
            yoy_revenue_growth = None
            qoq_profit_growth = None
            yoy_profit_growth = None
        
        # Get institutional data
        fii_dii_activity = detect_institutional_activity(volumes, closes)
        
        rsi = calculate_rsi(closes)
        macd = calculate_macd(closes)
        bb_position = calculate_bb_position(closes)
        vol_multiple = calculate_volume_multiple(volumes)
        trend = detect_trend(closes)
        
        # Calculate timeframe changes
        weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
        monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 else 0
        three_month_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 60 else 0
        
        return {
            'symbol': symbol,
            'price': price,
            'change': change,
            'weekly_change': weekly_change,
            'monthly_change': monthly_change,
            'three_month_change': three_month_change,
            'rsi': rsi,
            'macd': macd,
            'bb_position': bb_position,
            'vol_multiple': vol_multiple,
            'trend': trend,
            'closes': closes,
            'highs': highs,
            'lows': lows,
            'volumes': volumes,
            'fii_dii_score': fii_dii_activity,
            'market_cap': market_cap,
            'revenue_growth': revenue_growth,
            'profit_margin': profit_margin,
            'earnings_growth': earnings_growth,
            'pe_ratio': pe_ratio,
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'qoq_revenue_growth': qoq_revenue_growth,
            'yoy_revenue_growth': yoy_revenue_growth,
            'qoq_profit_growth': qoq_profit_growth,
            'yoy_profit_growth': yoy_profit_growth
        }
    except Exception as e:
        return None

def fetch_live_price(symbol):
    """Fetch only live price for auto-refresh (non-cached)"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except:
        return None

def detect_institutional_activity(volumes, closes):
    """Detect FII/DII activity patterns from volume and price action"""
    if len(volumes) < 20 or len(closes) < 20:
        return 0
    
    score = 0
    recent_days = 10
    
    for i in range(-recent_days, 0):
        vol_ratio = volumes[i] / np.mean(volumes[-60:]) if len(volumes) >= 60 else volumes[i] / np.mean(volumes)
        price_change = ((closes[i] - closes[i-1]) / closes[i-1]) * 100 if i > -len(closes) else 0
        
        if vol_ratio > 1.5 and price_change > 1:
            score += 2
        elif vol_ratio > 1.2 and price_change > 0.5:
            score += 1
        elif vol_ratio > 1.5 and price_change < -1:
            score -= 2
        elif vol_ratio > 1.2 and price_change < -0.5:
            score -= 1
    
    return score

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    if len(prices) < 26:
        return 0
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    return ema12 - ema26

def calculate_ema(prices, period):
    multiplier = 2 / (period + 1)
    ema = np.mean(prices[:period])
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_bb_position(prices, period=20):
    if len(prices) < period:
        return 50
    recent = prices[-period:]
    sma = np.mean(recent)
    std = np.std(recent)
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    current = prices[-1]
    if upper == lower:
        return 50
    position = ((current - lower) / (upper - lower)) * 100
    return max(0, min(100, position))

def calculate_volume_multiple(volumes):
    if len(volumes) < 20:
        return 1.0
    current = volumes[-1]
    avg20 = np.mean(volumes[-20:])
    if avg20 == 0:
        return 1.0
    return current / avg20

def detect_operator_activity(data):
    """Detect if stock shows signs of operator/manipulator activity"""
    closes = data['closes']
    volumes = data['volumes']
    highs = data['highs']
    lows = data['lows']
    
    warning_flags = []
    risk_score = 0
    
    if len(closes) < 20:
        return False, [], 0
    
    # 1. EXTREME VOLUME SPIKES
    recent_vols = volumes[-10:]
    avg_vol = np.mean(volumes[-60:]) if len(volumes) >= 60 else np.mean(volumes)
    max_recent_vol = np.max(recent_vols)
    
    if max_recent_vol > avg_vol * 5:
        warning_flags.append("🚨 EXTREME volume spike (>5x avg) - Possible pump")
        risk_score += 30
    elif max_recent_vol > avg_vol * 3:
        warning_flags.append("⚠️ High volume spike (>3x avg) - Monitor closely")
        risk_score += 15
    
    # 2. PRICE VOLATILITY
    recent_prices = closes[-10:]
    price_swings = []
    for i in range(1, len(recent_prices)):
        swing = abs((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]) * 100
        price_swings.append(swing)
    
    avg_swing = np.mean(price_swings) if price_swings else 0
    max_swing = np.max(price_swings) if price_swings else 0
    
    if max_swing > 8 and avg_swing > 3:
        warning_flags.append("🚨 Extreme volatility (>8% swings) - Operator activity likely")
        risk_score += 25
    elif max_swing > 5 and avg_swing > 2:
        warning_flags.append("⚠️ High volatility - Possible manipulation")
        risk_score += 12
    
    # 3. CIRCUIT FILTER HITS
    circuit_hits = 0
    for i in range(-20, 0):
        if i >= -len(closes):
            daily_change = abs((closes[i] - closes[i-1]) / closes[i-1]) * 100
            if daily_change > 9:
                circuit_hits += 1
    
    if circuit_hits >= 3:
        warning_flags.append("🚨 Multiple circuit hits - Highly manipulated")
        risk_score += 30
    elif circuit_hits >= 2:
        warning_flags.append("⚠️ Circuit hits detected - High risk")
        risk_score += 15
    
    is_operated = risk_score >= 40
    
    return is_operated, warning_flags, risk_score

def detect_trend(prices):
    if len(prices) < 5:
        return 'Sideways'
    recent = prices[-5:]
    ups = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
    if ups >= 4:
        return 'Strong Uptrend'
    elif ups >= 3:
        return 'Uptrend'
    elif ups <= 1:
        return 'Downtrend'
    else:
        return 'Sideways'

def analyze_stock(data, min_market_cap):
    """Analyze stock with ULTRA-STRICT fundamentals criteria"""
    if not data:
        return None
    
    price = data['price']
    change = data['change']
    rsi = data['rsi']
    macd = data['macd']
    bb = data['bb_position']
    vol = data['vol_multiple']
    trend = data['trend']
    closes = data['closes']
    
    # Market cap filter (in crores)
    market_cap = data['market_cap'] / 10000000 if data['market_cap'] else 0
    
    # Skip if below minimum market cap
    if market_cap < min_market_cap:
        return None
    
    # OPERATOR DETECTION
    is_operated, operator_flags, operator_risk = detect_operator_activity(data)
    
    # Calculate additional indicators
    weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
    monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 else 0
    three_month_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 60 else 0
    
    potential_rs = max(20, price * 0.10)
    potential_pct = (potential_rs / price) * 100
    
    score = 0
    criteria = []
    
    # CRITICAL: Operator penalty
    if is_operated:
        score -= 70
        criteria.append(f'🚨 OPERATOR DETECTED: Risk Score {operator_risk}/100 - AVOID [-70 pts]')
    elif operator_risk >= 30:
        score -= 40
        criteria.append(f'🚨 VERY HIGH RISK: Major manipulation signs (Risk: {operator_risk}/100) [-40 pts]')
    elif operator_risk >= 20:
        score -= 25
        criteria.append(f'⚠️ HIGH RISK: Manipulation signs detected (Risk: {operator_risk}/100) [-25 pts]')
    elif operator_risk >= 12:
        score -= 12
        criteria.append(f'⚠️ CAUTION: Some manipulation indicators (Risk: {operator_risk}/100) [-12 pts]')
    
    # 1. MARKET CAP QUALITY (15 pts) - NEW!
    if market_cap >= 50000:
        score += 15
        criteria.append(f'✅ Market Cap: Large Cap (₹{market_cap:.0f} Cr) [15 pts]')
    elif market_cap >= 20000:
        score += 12
        criteria.append(f'✅ Market Cap: Mid-Large Cap (₹{market_cap:.0f} Cr) [12 pts]')
    elif market_cap >= 10000:
        score += 10
        criteria.append(f'✅ Market Cap: Mid Cap (₹{market_cap:.0f} Cr) [10 pts]')
    elif market_cap >= 5000:
        score += 7
        criteria.append(f'⚠ Market Cap: Small-Mid Cap (₹{market_cap:.0f} Cr) [7 pts]')
    else:
        score += 0
        criteria.append(f'❌ Market Cap: Small Cap (₹{market_cap:.0f} Cr) [0 pts]')
    
    # 2. REVENUE GROWTH (25 pts) - NEW! STRICTEST CRITERIA
    yoy_rev = data['yoy_revenue_growth']
    qoq_rev = data['qoq_revenue_growth']
    
    if yoy_rev is not None and qoq_rev is not None:
        if yoy_rev >= 25 and qoq_rev >= 15:
            score += 25
            criteria.append(f'✅ Revenue: EXCEPTIONAL Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [25 pts]')
        elif yoy_rev >= 20 and qoq_rev >= 10:
            score += 22
            criteria.append(f'✅ Revenue: Excellent Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [22 pts]')
        elif yoy_rev >= 15 and qoq_rev >= 8:
            score += 18
            criteria.append(f'✅ Revenue: Strong Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [18 pts]')
        elif yoy_rev >= 10 and qoq_rev >= 5:
            score += 12
            criteria.append(f'⚠ Revenue: Good Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [12 pts]')
        elif yoy_rev >= 5:
            score += 5
            criteria.append(f'⚠ Revenue: Moderate Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [5 pts]')
        else:
            score += 0
            criteria.append(f'❌ Revenue: Weak/Negative Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [0 pts]')
    elif yoy_rev is not None:
        if yoy_rev >= 20:
            score += 20
            criteria.append(f'✅ Revenue: Strong YoY Growth ({yoy_rev:.1f}%) [20 pts]')
        elif yoy_rev >= 12:
            score += 15
            criteria.append(f'✅ Revenue: Good YoY Growth ({yoy_rev:.1f}%) [15 pts]')
        elif yoy_rev >= 5:
            score += 8
            criteria.append(f'⚠ Revenue: Moderate Growth ({yoy_rev:.1f}%) [8 pts]')
        else:
            score += 0
            criteria.append(f'❌ Revenue: Weak Growth ({yoy_rev:.1f}%) [0 pts]')
    else:
        score += 0
        criteria.append(f'❌ Revenue: Data not available [0 pts]')
    
    # 3. PROFIT GROWTH (25 pts) - NEW! STRICTEST CRITERIA
    yoy_profit = data['yoy_profit_growth']
    qoq_profit = data['qoq_profit_growth']
    profit_margin = data['profit_margin']
    
    if yoy_profit is not None and qoq_profit is not None:
        if yoy_profit >= 30 and qoq_profit >= 20:
            score += 25
            criteria.append(f'✅ Profit: EXCEPTIONAL Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [25 pts]')
        elif yoy_profit >= 25 and qoq_profit >= 15:
            score += 22
            criteria.append(f'✅ Profit: Excellent Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [22 pts]')
        elif yoy_profit >= 20 and qoq_profit >= 10:
            score += 18
            criteria.append(f'✅ Profit: Strong Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [18 pts]')
        elif yoy_profit >= 12 and qoq_profit >= 6:
            score += 12
            criteria.append(f'⚠ Profit: Good Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [12 pts]')
        elif yoy_profit >= 5:
            score += 5
            criteria.append(f'⚠ Profit: Moderate Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [5 pts]')
        else:
            score += 0
            criteria.append(f'❌ Profit: Weak/Negative Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [0 pts]')
    elif yoy_profit is not None:
        if yoy_profit >= 25:
            score += 20
            criteria.append(f'✅ Profit: Strong YoY Growth ({yoy_profit:.1f}%) [20 pts]')
        elif yoy_profit >= 15:
            score += 15
            criteria.append(f'✅ Profit: Good YoY Growth ({yoy_profit:.1f}%) [15 pts]')
        elif yoy_profit >= 8:
            score += 8
            criteria.append(f'⚠ Profit: Moderate Growth ({yoy_profit:.1f}%) [8 pts]')
        else:
            score += 0
            criteria.append(f'❌ Profit: Weak Growth ({yoy_profit:.1f}%) [0 pts]')
    else:
        score += 0
        criteria.append(f'❌ Profit: Data not available [0 pts]')
    
    # 4. PROFIT MARGIN (15 pts) - NEW!
    if profit_margin is not None:
        profit_margin_pct = profit_margin * 100
        if profit_margin_pct >= 20:
            score += 15
            criteria.append(f'✅ Profit Margin: Excellent ({profit_margin_pct:.1f}%) [15 pts]')
        elif profit_margin_pct >= 15:
            score += 12
            criteria.append(f'✅ Profit Margin: Very Good ({profit_margin_pct:.1f}%) [12 pts]')
        elif profit_margin_pct >= 10:
            score += 10
            criteria.append(f'✅ Profit Margin: Good ({profit_margin_pct:.1f}%) [10 pts]')
        elif profit_margin_pct >= 5:
            score += 5
            criteria.append(f'⚠ Profit Margin: Average ({profit_margin_pct:.1f}%) [5 pts]')
        else:
            score += 0
            criteria.append(f'❌ Profit Margin: Low ({profit_margin_pct:.1f}%) [0 pts]')
    else:
        score += 0
        criteria.append(f'❌ Profit Margin: Data not available [0 pts]')
    
    # 5. FII/DII ACTIVITY (20 pts)
    fii_score = data['fii_dii_score']
    if fii_score >= 15:
        score += 20
        criteria.append(f'✅ FII/DII: Strong Buying ({fii_score}) [20 pts]')
    elif fii_score >= 10:
        score += 15
        criteria.append(f'✅ FII/DII: Good Buying ({fii_score}) [15 pts]')
    elif fii_score >= 5:
        score += 10
        criteria.append(f'✅ FII/DII: Accumulation ({fii_score}) [10 pts]')
    elif fii_score >= 0:
        score += 5
        criteria.append(f'⚠ FII/DII: Neutral ({fii_score}) [5 pts]')
    else:
        score += 0
        criteria.append(f'❌ FII/DII: Selling ({fii_score}) [0 pts]')
    
    # 6. CONSOLIDATION (20 pts)
    if -2 <= weekly_change <= 0.3:
        score += 20
        criteria.append(f'✅ Consolidation: Perfect base ({weekly_change:+.1f}% weekly) [20 pts]')
    elif -3.5 <= weekly_change < -2:
        score += 18
        criteria.append(f'✅ Consolidation: Healthy pullback ({weekly_change:+.1f}% weekly) [18 pts]')
    elif 0.3 < weekly_change <= 1.5:
        score += 15
        criteria.append(f'✅ Consolidation: Early breakout ({weekly_change:+.1f}% weekly) [15 pts]')
    elif weekly_change > 4:
        score += 0
        criteria.append(f'❌ Already rallied ({weekly_change:+.1f}% weekly) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ Consolidation: Weak ({weekly_change:+.1f}% weekly) [5 pts]')
    
    # 7. RSI (20 pts)
    if 32 <= rsi <= 38:
        score += 20
        criteria.append(f'✅ RSI: Perfect oversold entry ({rsi:.0f}) [20 pts]')
    elif 38 < rsi <= 45:
        score += 17
        criteria.append(f'✅ RSI: Building momentum ({rsi:.0f}) [17 pts]')
    elif 45 < rsi <= 50:
        score += 12
        criteria.append(f'✅ RSI: Early momentum ({rsi:.0f}) [12 pts]')
    elif 50 < rsi <= 55:
        score += 8
        criteria.append(f'⚠ RSI: Neutral ({rsi:.0f}) [8 pts]')
    elif rsi > 62:
        score += 0
        criteria.append(f'❌ RSI: Overbought ({rsi:.0f}) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ RSI: Moderate ({rsi:.0f}) [5 pts]')
    
    # 8. MACD (15 pts)
    if -1 <= macd <= 1:
        score += 15
        criteria.append(f'✅ MACD: Perfect crossover ({macd:.1f}) [15 pts]')
    elif 1 < macd <= 3:
        score += 12
        criteria.append(f'✅ MACD: Early bullish ({macd:.1f}) [12 pts]')
    elif -3 <= macd < -1:
        score += 10
        criteria.append(f'✅ MACD: About to turn ({macd:.1f}) [10 pts]')
    elif macd > 6:
        score += 0
        criteria.append(f'❌ MACD: Extended ({macd:.1f}) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ MACD: Weak ({macd:.1f}) [5 pts]')
    
    # 9. BOLLINGER BANDS (15 pts)
    if 8 <= bb <= 20:
        score += 15
        criteria.append(f'✅ BB: Lower band bounce ({bb:.0f}%) [15 pts]')
    elif 20 < bb <= 30:
        score += 12
        criteria.append(f'✅ BB: Below middle ({bb:.0f}%) [12 pts]')
    elif 30 < bb <= 45:
        score += 8
        criteria.append(f'⚠ BB: Middle zone ({bb:.0f}%) [8 pts]')
    elif bb > 65:
        score += 0
        criteria.append(f'❌ BB: Upper band ({bb:.0f}%) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ BB: Neutral ({bb:.0f}%) [5 pts]')
    
    # 10. VOLUME (15 pts)
    if 1.3 <= vol <= 1.8:
        score += 15
        criteria.append(f'✅ Volume: Perfect accumulation ({vol:.1f}x) [15 pts]')
    elif 1.8 < vol <= 2.2:
        score += 12
        criteria.append(f'✅ Volume: Building interest ({vol:.1f}x) [12 pts]')
    elif vol > 2.8:
        score += 5
        criteria.append(f'⚠ Volume: Too high ({vol:.1f}x) [5 pts]')
    elif 1.0 <= vol < 1.3:
        score += 7
        criteria.append(f'⚠ Volume: Average ({vol:.1f}x) [7 pts]')
    else:
        score += 0
        criteria.append(f'❌ Volume: Too low ({vol:.1f}x) [0 pts]')
    
    # 11. TODAY'S PRICE (10 pts)
    if -1.5 <= change <= 0.3:
        score += 10
        criteria.append(f'✅ Today: Perfect entry ({change:+.1f}%) [10 pts]')
    elif 0.3 < change <= 1.2:
        score += 8
        criteria.append(f'✅ Today: Early move ({change:+.1f}%) [8 pts]')
    elif -2.5 <= change < -1.5:
        score += 7
        criteria.append(f'⚠ Today: Dip ({change:+.1f}%) [7 pts]')
    elif change > 2.5:
        score += 0
        criteria.append(f'❌ Today: Already rallied ({change:+.1f}%) [0 pts]')
    else:
        score += 4
        criteria.append(f'⚠ Today: Moderate ({change:+.1f}%) [4 pts]')
    
    # 12. MONTHLY TREND (10 pts)
    if -8 <= monthly_change <= -2:
        score += 10
        criteria.append(f'✅ Monthly: Recovering from dip ({monthly_change:+.1f}%) [10 pts]')
    elif -2 < monthly_change <= 2:
        score += 8
        criteria.append(f'✅ Monthly: Base building ({monthly_change:+.1f}%) [8 pts]')
    elif 2 < monthly_change <= 6:
        score += 5
        criteria.append(f'⚠ Monthly: Moderate gain ({monthly_change:+.1f}%) [5 pts]')
    elif monthly_change > 10:
        score += 0
        criteria.append(f'❌ Monthly: Extended ({monthly_change:+.1f}%) [0 pts]')
    else:
        score += 3
        criteria.append(f'⚠ Monthly: Weak ({monthly_change:+.1f}%) [3 pts]')
    
    # 13. 3-MONTH PERFORMANCE (10 pts)
    if -15 <= three_month_change <= -5:
        score += 10
        criteria.append(f'✅ 3-Month: Perfect correction ({three_month_change:+.1f}%) [10 pts]')
    elif -5 < three_month_change <= 5:
        score += 8
        criteria.append(f'✅ 3-Month: Sideways base ({three_month_change:+.1f}%) [8 pts]')
    elif 5 < three_month_change <= 15:
        score += 5
        criteria.append(f'⚠ 3-Month: Moderate rise ({three_month_change:+.1f}%) [5 pts]')
    elif three_month_change > 25:
        score += 0
        criteria.append(f'❌ 3-Month: Overextended ({three_month_change:+.1f}%) [0 pts]')
    else:
        score += 3
        criteria.append(f'⚠ 3-Month: Weak ({three_month_change:+.1f}%) [3 pts]')
    
    # 14. UPSIDE POTENTIAL (10 pts)
    if potential_pct >= 12:
        score += 10
        criteria.append(f'✅ Upside: Excellent ({potential_pct:.1f}%) [10 pts]')
    elif potential_pct >= 10:
        score += 8
        criteria.append(f'✅ Upside: Very Good ({potential_pct:.1f}%) [8 pts]')
    elif potential_pct >= 8:
        score += 5
        criteria.append(f'⚠ Upside: Good ({potential_pct:.1f}%) [5 pts]')
    else:
        score += 0
        criteria.append(f'❌ Upside: Low ({potential_pct:.1f}%) [0 pts]')
    
    # Rating based on ULTRA-STRICT criteria
    if is_operated:
        status = '🚨 OPERATED - AVOID'
        rating = 'Operated - Avoid'
    elif score >= 180:
        status = '🌟 EXCEPTIONAL BUY'
        rating = 'Exceptional Buy'
    elif score >= 160:
        status = '🚀 PRIME BUY'
        rating = 'Prime Buy'
    elif score >= 140:
        status = '💎 EXCELLENT BUY'
        rating = 'Excellent Buy'
    elif score >= 120:
        status = '✅ STRONG BUY'
        rating = 'Strong Buy'
    elif score >= 100:
        status = '👍 GOOD BUY'
        rating = 'Good Buy'
    elif score >= 80:
        status = '📋 WATCHLIST'
        rating = 'Watchlist'
    else:
        status = '❌ SKIP'
        rating = 'Skip'
    
    qualified = score >= 140 and not is_operated
    met_count = len([c for c in criteria if '✅' in c])
    
    return {
        'symbol': data['symbol'],
        'price': price,
        'change': change,
        'weekly_change': weekly_change,
        'monthly_change': monthly_change,
        'three_month_change': three_month_change,
        'potential_rs': potential_rs,
        'potential_pct': potential_pct,
        'rsi': rsi,
        'macd': macd,
        'bb': bb,
        'vol': vol,
        'trend': trend,
        'score': score,
        'qualified': qualified,
        'status': status,
        'rating': rating,
        'criteria': criteria,
        'met_count': met_count,
        'sector': SECTOR_MAP.get(data['symbol'], 'Other'),
        'is_operated': is_operated,
        'operator_risk': operator_risk,
        'operator_flags': operator_flags,
        'market_cap': market_cap,
        'yoy_revenue_growth': yoy_rev,
        'qoq_revenue_growth': qoq_rev,
        'yoy_profit_growth': yoy_profit,
        'qoq_profit_growth': qoq_profit,
        'profit_margin': profit_margin * 100 if profit_margin else None
    }

# Main App
st.markdown('<p class="main-header">🎯 Indian Stock Scout - ULTRA-STRICT with Fundamentals</p>', unsafe_allow_html=True)
st.markdown("*Only stocks with EXCEPTIONAL fundamentals + technicals qualify*")

# Sidebar
st.sidebar.header("⚙ Scanner Configuration")

scan_mode = st.sidebar.radio("Scan Mode", 
    ["Quick Scan (50 stocks)", "Full Scan (100+ stocks)", "Custom List"])

if scan_mode == "Quick Scan (50 stocks)":
    stocks_to_scan = NSE_STOCKS[:50]
elif scan_mode == "Full Scan (100+ stocks)":
    stocks_to_scan = NSE_STOCKS
else:
    custom_input = st.sidebar.text_area("Enter NSE symbols (one per line)", 
        "RELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK", height=150)
    stocks_to_scan = [s.strip().upper() for s in custom_input.split('\n') if s.strip()]

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Market Cap Filter")
min_market_cap = st.sidebar.slider("Minimum Market Cap (₹ Crores)", 
    0, 100000, 5000, 1000,
    help="Filter stocks by minimum market capitalization")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 ULTRA-STRICT Criteria")
st.sidebar.info("""
*Only top 1-3% qualify!*

**TOTAL: 250 Points**

**Fundamentals (80 pts):**
1. Market Cap (15 pts)
2. Revenue Growth (25 pts)
   - YoY ≥20%, QoQ ≥10%
3. Profit Growth (25 pts)
   - YoY ≥25%, QoQ ≥15%
4. Profit Margin (15 pts)
   - ≥15% excellent

**Technicals (170 pts):**
5. FII/DII (20 pts)
6. Consolidation (20 pts)
7. RSI (20 pts): 32-38
8. MACD (15 pts): -1 to +1
9. BB (15 pts): 8-20%
10. Volume (15 pts): 1.3-1.8x
11. Today (10 pts)
12. Monthly (10 pts)
13. 3-Month (10 pts)
14. Upside (10 pts): ≥12%

**Qualification:**
- Exceptional: ≥180 pts
- Prime: 160-179 pts
- Excellent: 140-159 pts ✅
- Strong: 120-139 pts
- Below 140: Not qualified

**Penalties:**
- Operated: -70 pts
- High Risk: -25 to -40 pts
""")

st.sidebar.markdown("---")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

if st.sidebar.button("🚀 FIND EXCEPTIONAL STOCKS", type="primary", use_container_width=True):
    st.markdown("---")
    st.subheader("📊 Scanning with Fundamental + Technical Analysis...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_placeholder = st.empty()
    
    results = []
    total = len(stocks_to_scan)
    failed = 0
    filtered_out = 0
    
    for idx, symbol in enumerate(stocks_to_scan):
        status_text.info(f"📊 Analyzing *{symbol}*... ({idx+1}/{total})")
        
        data = fetch_stock_data(symbol)
        if data:
            analysis = analyze_stock(data, min_market_cap)
            if analysis:
                results.append(analysis)
            else:
                filtered_out += 1
        else:
            failed += 1
        
        progress = (idx + 1) / total
        progress_bar.progress(progress)
        
        if (idx + 1) % 10 == 0 or idx == total - 1:
            valid_results_count = len([r for r in results if r is not None])
            qualified_count = len([r for r in results if r is not None and r['qualified']])
            stats_placeholder.info(f"✅ Valid: {valid_results_count} | Qualified (≥140): {qualified_count} | Filtered: {filtered_out} | Failed: {failed}")
        
        time.sleep(0.2)
    
    # Filter out None results before storing
    results = [r for r in results if r is not None]
    st.session_state.scan_results = results
    st.session_state.scan_timestamp = datetime.now()
    
    status_text.success(f"✅ Scan complete! Found {len(results)} stocks meeting market cap criteria")
    time.sleep(1)
    status_text.empty()
    stats_placeholder.empty()
    progress_bar.empty()
    
    st.rerun()

# Display results
if st.session_state.scan_results:
    results = st.session_state.scan_results
    scan_time = st.session_state.scan_timestamp
    
    st.markdown("---")
    
    # Auto-refresh toggle
    col_refresh1, col_refresh2, col_refresh3 = st.columns([2, 2, 6])
    
    with col_refresh1:
        auto_refresh = st.checkbox("🔄 Auto-refresh prices", value=False, 
                                   help="Continuously update prices every 30 seconds without resetting")
    
    with col_refresh2:
        if 'last_refresh' in st.session_state:
            seconds_ago = int((datetime.now() - st.session_state.last_refresh).total_seconds())
            st.caption(f"📡 Updated {seconds_ago}s ago")
        else:
            st.caption("📡 Not refreshed yet")
    
    with col_refresh3:
        if auto_refresh:
            if st.button("⏸️ Pause Refresh"):
                st.session_state.auto_refresh_paused = True
                st.rerun()
    
    st.subheader(f"📈 Exceptional Stock Opportunities")
    st.caption(f"Initial scan: {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh logic - NON-BLOCKING
    if auto_refresh and not st.session_state.get('auto_refresh_paused', False):
        # Initialize refresh counter
        if 'refresh_counter' not in st.session_state:
            st.session_state.refresh_counter = 0
            st.session_state.last_refresh = datetime.now()
        
        # Check if 30 seconds have passed
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        
        if time_since_refresh >= 30:
            # Update prices in background
            with st.spinner("🔄 Refreshing live prices..."):
                updated_count = 0
                for result in results:
                    new_price = fetch_live_price(result['symbol'])
                    if new_price and new_price != result['price']:
                        prev_price = result['price']
                        result['price'] = new_price
                        result['change'] = ((new_price - prev_price) / prev_price) * 100
                        updated_count += 1
                
                st.session_state.last_refresh = datetime.now()
                st.session_state.refresh_counter += 1
                
                if updated_count > 0:
                    st.toast(f"✅ Updated {updated_count} prices", icon="🔄")
            
            # Force re-render
            time.sleep(1)
            st.rerun()
        else:
            # Schedule next refresh
            remaining = 30 - time_since_refresh
            st.info(f"⏱️ Next refresh in {int(remaining)} seconds...")
            time.sleep(5)  # Check every 5 seconds
            st.rerun()
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'Symbol': r['symbol'],
        'Price (₹)': r['price'],
        'Today (%)': r['change'],
        'Weekly (%)': r['weekly_change'],
        'Monthly (%)': r['monthly_change'],
        '3M (%)': r['three_month_change'],
        'Market Cap (₹Cr)': r['market_cap'],
        'Rev YoY (%)': r['yoy_revenue_growth'],
        'Rev QoQ (%)': r['qoq_revenue_growth'],
        'Profit YoY (%)': r['yoy_profit_growth'],
        'Profit QoQ (%)': r['qoq_profit_growth'],
        'Margin (%)': r['profit_margin'],
        'RSI': r['rsi'],
        'MACD': r['macd'],
        'BB (%)': r['bb'],
        'Vol': f"{r['vol']:.1f}x",
        'Score': r['score'],
        'Rating': r['rating'],
        'Status': r['status'],
        'Sector': r['sector'],
        'Operated': '🚨 YES' if r['is_operated'] else '✅ Safe',
        'Risk': r['operator_risk']
    } for r in results])
    
    # Statistics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    operated_stocks = df[df['Operated'] == '🚨 YES']
    safe_stocks = df[df['Operated'] == '✅ Safe']
    exceptional = df[(df['Score'] >= 180) & (df['Operated'] == '✅ Safe')]
    prime = df[(df['Score'] >= 160) & (df['Score'] < 180) & (df['Operated'] == '✅ Safe')]
    excellent = df[(df['Score'] >= 140) & (df['Score'] < 160) & (df['Operated'] == '✅ Safe')]
    strong = df[(df['Score'] >= 120) & (df['Score'] < 140) & (df['Operated'] == '✅ Safe')]
    
    col1.metric("Total Scanned", len(df))
    col2.metric("🚨 Operated", len(operated_stocks))
    col3.metric("🌟 Exceptional (≥180)", len(exceptional))
    col4.metric("🚀 Prime (160-179)", len(prime))
    col5.metric("💎 Excellent (140-159)", len(excellent))
    col6.metric("✅ Strong (120-139)", len(strong))
    
    # Qualification summary
    qualified_total = len(exceptional) + len(prime) + len(excellent)
    st.success(f"""
    **🎯 ULTRA-STRICT RESULTS:** Only **{qualified_total}** stocks qualified (Score ≥140 + Safe) out of {len(df)}.
    That's the top **{(qualified_total/len(df)*100) if len(df) > 0 else 0:.1f}%** - truly exceptional opportunities with strong fundamentals!
    """)
    
    st.markdown("---")
    
    # Filtering
    st.subheader("🔍 Filter Results")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        rating_filter = st.selectbox("Rating", 
            ["All", "Exceptional Buy", "Prime Buy", "Excellent Buy", "Strong Buy", "Good Buy", "Watchlist", "Skip"])
    
    with filter_col2:
        safety_filter = st.selectbox("Safety", 
            ["All", "✅ Safe Only", "🚨 Operated Only"])
    
    with filter_col3:
        sector_filter = st.selectbox("Sector", 
            ["All"] + sorted(df['Sector'].unique().tolist()))
    
    with filter_col4:
        min_score_filter = st.number_input("Min Score", 0, 250, 140, 10,
                                          help="Default: 140 (Qualified)")
    
    # Apply filters
    filtered_df = df.copy()
    
    if rating_filter != "All":
        filtered_df = filtered_df[filtered_df['Rating'] == rating_filter]
    
    if safety_filter == "✅ Safe Only":
        filtered_df = filtered_df[filtered_df['Operated'] == '✅ Safe']
    elif safety_filter == "🚨 Operated Only":
        filtered_df = filtered_df[filtered_df['Operated'] == '🚨 YES']
    
    if sector_filter != "All":
        filtered_df = filtered_df[filtered_df['Sector'] == sector_filter]
    
    filtered_df = filtered_df[filtered_df['Score'] >= min_score_filter]
    
    st.info(f"📊 Showing *{len(filtered_df)}* stocks (filtered from {len(df)} total)")
    
    # Display table
    st.subheader("📋 Stock Analysis Table")
    
    def highlight_rating(row):
        if row['Operated'] == '🚨 YES':
            return ['background-color: #ff6b6b; color: white; font-weight: bold'] * len(row)
        elif row['Score'] >= 180:
            return ['background-color: #00e676; color: black; font-weight: bold'] * len(row)
        elif row['Score'] >= 160:
            return ['background-color: #69f0ae; font-weight: bold'] * len(row)
        elif row['Score'] >= 140:
            return ['background-color: #b9f6ca; font-weight: bold'] * len(row)
        elif row['Score'] >= 120:
            return ['background-color: #e1f5fe'] * len(row)
        elif row['Score'] >= 100:
            return ['background-color: #fff9c4'] * len(row)
        else:
            return ['background-color: #ffebee'] * len(row)
    
    styled_df = filtered_df.style.apply(highlight_rating, axis=1)\
        .format({
            'Price (₹)': '₹{:.2f}',
            'Today (%)': '{:+.2f}%',
            'Weekly (%)': '{:+.2f}%',
            'Monthly (%)': '{:+.2f}%',
            '3M (%)': '{:+.2f}%',
            'Market Cap (₹Cr)': '₹{:.0f}',
            'Rev YoY (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
            'Rev QoQ (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
            'Profit YoY (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
            'Profit QoQ (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
            'Margin (%)': lambda x: f'{x:.1f}%' if pd.notna(x) else 'N/A',
            'RSI': '{:.1f}',
            'MACD': '{:.2f}',
            'BB (%)': '{:.0f}%'
        })
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    # Detailed view
    st.markdown("---")
    st.subheader("🔍 Detailed Stock Analysis")
    
    if len(filtered_df) > 0:
        selected_symbol = st.selectbox("Select stock for details", filtered_df['Symbol'].tolist())
        selected_result = next((r for r in results if r['symbol'] == selected_symbol), None)
        
        if selected_result:
            st.markdown(f"### {selected_symbol} - {selected_result['status']}")
            
            if selected_result['is_operated']:
                st.error(f"🚨 **OPERATOR DETECTED** - Risk: {selected_result['operator_risk']}/100")
                for flag in selected_result['operator_flags']:
                    st.warning(flag)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Score", selected_result['score'])
            col2.metric("Price", f"₹{selected_result['price']:.2f}")
            col3.metric("Market Cap", f"₹{selected_result['market_cap']:.0f}Cr")
            col4.metric("Rev YoY", f"{selected_result['yoy_revenue_growth']:+.1f}%" if selected_result['yoy_revenue_growth'] else "N/A")
            col5.metric("Profit YoY", f"{selected_result['yoy_profit_growth']:+.1f}%" if selected_result['yoy_profit_growth'] else "N/A")
            
            st.markdown("#### Detailed Scoring Breakdown")
            for criterion in selected_result['criteria']:
                if '🚨' in criterion:
                    st.error(criterion)
                elif '✅' in criterion:
                    st.success(criterion)
                elif '⚠' in criterion:
                    st.warning(criterion)
                else:
                    st.error(criterion)
    
    # Download
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered CSV",
            csv,
            f"ultra_strict_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        all_csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download All Results CSV",
            all_csv,
            f"ultra_strict_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )

else:
    st.info("👈 Configure and click 'FIND EXCEPTIONAL STOCKS' to start")
    
    st.markdown("---")
    st.subheader("🎯 Why ULTRA-STRICT Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **NEW: Fundamental Analysis Added!**
        
        **4 Fundamental Criteria (80 pts):**
        1. **Market Cap (15 pts)**
           - Filters quality companies
           - Large cap = More stable
        
        2. **Revenue Growth (25 pts)**
           - YoY ≥20% + QoQ ≥10% = 22+ pts
           - Must show consistent growth
        
        3. **Profit Growth (25 pts)**
           - YoY ≥25% + QoQ ≥15% = 22+ pts
           - Even stricter than revenue
        
        4. **Profit Margin (15 pts)**
           - ≥15% = Excellent (12+ pts)
           - Shows business quality
        
        **Result:** Only 1-3% stocks pass ALL criteria!
        """)
    
    with col2:
        st.markdown("""
        **Complete 14-Point System:**
        
        **Fundamentals (80 pts)**
        - Market cap quality
        - Revenue acceleration
        - Profit growth momentum
        - Margin efficiency
        
        **Technicals (170 pts)**
        - Institutional buying
        - Perfect consolidation
        - Oversold RSI (32-38)
        - MACD crossover
        - Lower BB bounce
        - Accumulation volume
        - Entry timing
        - Trend confirmation
        
        **Safety (Penalties)**
        - Operator detection: -70 pts
        - Risk penalties: up to -40 pts
        """)
    
    st.markdown("---")
    st.error("""
    **⚠️ ULTRA-STRICT = TOP 1-3% ONLY:**
    
    With fundamentals + technicals:
    - **1-3 stocks** out of 100 qualify (1-3%)
    - Must have ≥140 points (Perfect fundamentals + technicals)
    - **0-1 exceptional** (score ≥180)
    - **1-2 excellent** (score 140-179)
    
    **Bottom Line:** We find the absolute BEST opportunities - companies with explosive growth + perfect technical setup!
    """)

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;'>
<p><strong>Ultra-Strict Scanner with Fundamentals</strong> | Top 1-3% Only</p>
<p style='font-size:0.85rem;'>⚠ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
