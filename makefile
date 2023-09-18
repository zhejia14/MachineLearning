data:points30.py points2000.py miss.py
	python points30.py
	python points2000.py
	python miss.py
pla30:pla.py points30.txt
	python pla.py points30.txt
pla2000:pla.py points2000.txt
	python pla.py points2000.txt
plaMiss:pla.py miss.txt
	python pla.py miss.txt
pok30:pocket.py points30.txt
	python pocket.py points30.txt
pok2000:pocket.py points2000.txt
	python pocket.py points2000.txt
pokMiss:pocket.py miss.txt
	python pocket.py miss.txt
clean:
	rm points30.txt
	rm points2000.txt
	rm miss.txt
