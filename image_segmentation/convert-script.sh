for filename in ./background-set/*ppm
do
	A=`cut -d'.' -f1 <<< $filname`
	A="$A.ppm"
	convert $filename $A
	rm $filename
done