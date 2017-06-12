# Austin Voecks
# CS 570, a4
# makefile

JAVAC=javac -cp .:commons-math3-3.6.1.jar
sources = $(wildcard *.java)
classes = $(sources:.java=.class)

all: $(classes)

clean:
		@rm -f *.class
		@echo Done
	
%.class : %.java
		$(JAVAC) $<
