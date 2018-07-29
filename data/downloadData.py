from bs4 import BeautifulSoup
import urllib

if __name__ == "__main__":
	url = "http://nrvnqsr.wikia.com/wiki/FSN_Fate_Day_01_(EN)"
	output = []
	for x in xrange(3,15):
		html = urllib.urlopen(url)
		bs = BeautifulSoup(html, "html.parser")

		text = bs.findAll("div", {"id":"mw-content-text"})[0].findChildren("p", recursive=False)
		text = [unicode(a.get_text().encode("ascii", "ignore"), "utf-8").replace("\n\n", "\n").replace("\"\"", "") for a in text]
		
		output.append("".join(text))

	print "\n".join(output)
