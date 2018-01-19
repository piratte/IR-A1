download:
		mkdir obj
		wget https://www.dropbox.com/s/wgoast2uyr1xi6n/pickles.tgz
		tar -xvf pickles.tgz -C obj
		python3 solution/dictkeys-tolower.py obj/idf
		rm pickles.tgz

makerunable:
		chmod +x run

all:    download makerunable

default:    all

clean:
	rm -rf obj