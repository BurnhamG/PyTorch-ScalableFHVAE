all: kaldi
.PHONY: kaldi

kaldi:
	git clone https://github.com/kaldi-asr/kaldi.git kaldi
	cd kaldi/tools; ./extras/check_dependencies.sh; $(MAKE) all -j 4
	cd kaldi/src; ./configure --shared; $(MAKE) depend -j 8; $(MAKE) all -j 4

clean:
	rm -rf kaldi
