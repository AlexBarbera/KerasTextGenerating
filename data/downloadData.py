from bs4 import BeautifulSoup
import urllib

if __name__ == "__main__":
	url = "http://nrvnqsr.wikia.com/wiki/FSN_HF_Day_"#1_(EN)"
	output = []
	for i in xrange(4,17):
		html = None

		if i > 9:
			html = urllib.urlopen(url + str(i) + "_(EN)")
		else:
			html = urllib.urlopen(url + "0" + str(i) + "_(EN)")

		bs = BeautifulSoup(html, "html.parser")

		text = bs.findAll("div", {"id":"mw-content-text"})[0].findChildren("p", recursive=False)
		text = [unicode(a.get_text().encode("ascii", "ignore"), "utf-8").replace("(?<!.)\n", "\n").replace("\"\"", "").replace("\n\s+", "\n").replace(',\n', ", ") for a in text if "[" not in a.get_text()]

		output.append("".join(text))

	print "\n".join(output)
