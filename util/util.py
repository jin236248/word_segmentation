from datetime import datetime

def report_time(begin): 
    print(' time: ' + str(datetime.now() - begin)[:7], end=' ')
    
def show_progress(i, onetenth):
    if i % onetenth == onetenth - 1:print(i // onetenth, end='')

def preprocessing(txt: str, remove_space: bool = True) -> str:
    # from pythainlp
    
    SEPARATOR = "|"
    SURROUNDING_SEPS_RX = re.compile("{sep}? ?{sep}$".format(sep=re.escape(SEPARATOR)))
    MULTIPLE_SEPS_RX = re.compile("{sep}+".format(sep=re.escape(SEPARATOR)))
    TAG_RX = re.compile("<\/?[A-Z]+>")
    TAILING_SEP_RX = re.compile("{sep}$".format(sep=re.escape(SEPARATOR)))

    txt = re.sub(SURROUNDING_SEPS_RX, "", txt)
    if remove_space:
        txt = re.sub("\s+", "", txt)
    txt = re.sub(MULTIPLE_SEPS_RX, SEPARATOR, txt)
    txt = re.sub(TAG_RX, "", txt)
    txt = re.sub(TAILING_SEP_RX, "", txt).strip()

    return txt