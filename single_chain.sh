#!/bin/bash
# filename single_chain.sh
# Usage: single_chain.sh InitialPassword NumberOfLinks
# Passwords are always reduced to UUnnllU form
# Output approx 100 links/sec
# actual hash
# 97041cb4 919434a3 9cc3d21e 66b68487 4e5482bf 6d95a011 4077992c a5a95236
# test hash (hex)
# 
# reduction ZA90azI

TESTHASH=000000190000000900190000000700080009000a000b000c000d000effffffff
UCA=0x41
LCA=0x61
NUM=0x31
CHAIN_IDX=0
INITIAL=AB12cdE
LINKS=8
MASK=0xffff

if [[ $# -ne 2 ]]
then
	printf "Usage: single_chain AB12cdE nLinks\n"
	exit 1
else
	INITIAL=$1
	LINKS=$2
fi
echo -e "\nINITIAL: $INITIAL"

while [[ $CHAIN_IDX -lt $LINKS ]]
do
	HCOPY=`echo -n $INITIAL | sha256sum | awk '{print($1)}'`
	echo "Working: $INITIAL"
	#HCOPY=$TESTHASH
	echo -e "Hash: $HCOPY"

	Z=${HCOPY:0:8}
	#echo "Word: $Z"
	HCOPY=${HCOPY:8}
	# convert hex to dec
	Z=`echo $((0x$Z))`
	#echo "decimal Z:$Z"
	Z=$(( Z + CHAIN_IDX ))
	#echo "Plus chain idx:$Z"

	# ----- Get two upper case letters -----
	B0=$((Z & $MASK ))
	#echo -e "\nmask to 9 bits :$B0"
	#printf "B0: \x$(printf %x $B0)\n"
	B0=$(( B0 % 26 ))
	#echo "Virtual table index B1:$B0"
	# 'A' ascii 65 dec or 0x41
	B0=$(( B0 + 65 ))
	# index A is 0
	#echo "Ascii char B0:$B0"
	#printf "B0: \x$(printf %x $B0)\n"

	B0=`printf "\x$(printf %x $B0)"`

	B1=$((  (Z >> 16) & $MASK ))
	#echo "shift right 16 and mask to 9 bits :$B1"
	B1=$(( B1 % 26 ))
	#echo "Virtual table index B1:$B1"
	# 'A' ascii 65 dec or 0x41
	B1=$(( B1 + 65 ))
	# index A is 0
	#echo "Ascii char B1:$B1"
	#printf "B1: \x$(printf %x $B1)\n"

	B1=`printf "\x$(printf %x $B1)"`

	# ----- Get two numbers 0-9 -----
	Z=${HCOPY:0:8}
	#echo "Word: $Z"
	HCOPY=${HCOPY:8}
	# convert hex to dec
	Z=`echo $((0x$Z))`
	#echo "decimal Z:$Z"
	Z=$(( Z + CHAIN_IDX ))
	#echo "Plus chain idx:$Z"

	B2=$((Z & $MASK ))
	#echo -e "\nmask to 9 bits :$B2"
	B2=$(( B2 % 10 ))
	#echo "Virtual table index B2:$B2"
	# index 0 is 48
	B2=$(( B2 + 48 ))
	#echo "Ascii char B2:$B2"
	#printf "B2: \x$(printf %x $B2)\n"
	 
	B2=`printf "\x$(printf %x $B2)"`

	B3=$((  (Z >> 16) & $MASK ))
	#echo "shift right 16 and mask to 9 bits :$B3"
	B3=$(( B3 % 10 ))
	#echo "Virtual table index B3:$B3"
	# index 0 is 48
	B3=$(( B3 + 48 ))
	#echo "Ascii char B3:$B3"
	#printf "B3: \x$(printf %x $B3)\n"

	B3=`printf "\x$(printf %x $B3)"`

	# ----- Get two lower case letters -----
	Z=${HCOPY:0:8}
	#echo "Word: $Z"
	HCOPY=${HCOPY:8}
	# convert hex to dec
	Z=`echo $((0x$Z))`
	#echo "decimal Z:$Z"
	Z=$(( Z + CHAIN_IDX ))
	#echo "Plus chain idx:$Z"

	B4=$((Z & $MASK ))
	#echo -e "\nmask to 9 bits :$B4"
	B4=$(( B4 % 26 ))
	#echo "Virtual table index B4:$B4"
	# index a is 97
	B4=$(( B4 + 97 ))
	#echo "Ascii char B4:$B4"
	#printf "B4: \x$(printf %x $B4)\n"

	B4=`printf "\x$(printf %x $B4)"`
	 
	B5=$((  (Z >> 16) & $MASK ))
	#echo "shift right 16 and mask to 9 bits :$B5"
	B5=$(( B5 % 26 ))
	#echo "Virtual table index B5:$B5"
	# index a is 97
	B5=$(( B5 + 97 ))	
	#echo "Ascii char B5:$B5"
	#printf "B5: \x$(printf %x $B5)\n"

	B5=`printf "\x$(printf %x $B5)"`

	# ----- Get one upper case letters -----
	Z=${HCOPY:0:8}
	#echo "Word: $Z"
	HCOPY=${HCOPY:8}
	# convert hex to dec
	Z=`echo $((0x$Z))`
	#echo "decimal Z:$Z"
	Z=$(( Z + CHAIN_IDX ))
	#echo "Plus chain idx:$Z"

	B6=$((Z & $MASK ))
	#echo -e "\nmask to 9 bits :$B6"
	B6=$(( B6 % 26 ))
	#echo "Virtual table index B6:$B6"
	B6=$(( B6 + 65 ))
	#echo "Ascii char B6:$B6"
	#printf "B6: \x$(printf %x $B6)\n"
	
	B6=`printf "\x$(printf %x $B6)"`

	# ----- Print Reduced Hash -----
	echo "REDUCED: $B0$B1$B2$B3$B4$B5$B6"
	# ----- Set up for next pass -----
	INITIAL=$B0$B1$B2$B3$B4$B5$B6
	CHAIN_IDX=$((CHAIN_IDX+1))
done
echo -e "\n"

