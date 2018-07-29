from bs4 import BeautifulSoup
import urllib

if __name__ == "__main__":
	url = "http://nrvnqsr.wikia.com/wiki/FSN_Fate_Day_0"#1_(EN)"
	output = []
	for i in xrange(1,16):
		html = urllib.urlopen(url + str(i) + "_(EN)")
		bs = BeautifulSoup(html, "html.parser")

		text = bs.findAll("div", {"id":"mw-content-text"})[0].findChildren("p", recursive=False)
		text = [unicode(a.get_text().encode("ascii", "ignore"), "utf-8").replace("(?<!.)\n", "\n").replace("\"\"", "").replace(',\n', ",").replace("\n ", "\n") for a in text]
		text = "".join(text)

		output.append(text)

	print "\n".join(output)
