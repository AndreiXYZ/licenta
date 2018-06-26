for filename in /background_set_multi/*ppm 
do
	A=`cut -d'.' -f1 <<< $filename`
	A="$A.ppm"
	convert $filename $A
    echo Converted $filename to $A
	rm $filename
done
