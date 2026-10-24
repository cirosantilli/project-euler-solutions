import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot002Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root2Candidates0 : List Nat := [743, 1871, 3083, 4547, 5849, 6329, 7331, 8171, 8423, 8681, 8747, 14591, 20807, 21023]
set_option maxRecDepth 8192 in
theorem root2Candidates0_checked : candidates edge 4 11 23 root2Tail1 seed.bound = root2Candidates0 := by
  unfold candidates
  have row : edge 23 = (SortedNatTree.contains FastRows.row23) := by
    change Data.row23.contains = (SortedNatTree.contains FastRows.row23)
    exact FastRows.row23_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child0 : search edge 3 [11, 23] root2Candidates0 34 seed = seed := by decide
theorem root2Step0 : scan edge (search edge 3) 4 [11] 11 root2Tail0 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail1 seed := by
  rw [root2Tail0, scan, if_neg (by decide), root2Candidates0_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail1
    (if 3 ≤ root2Candidates0.length then search edge 3 [11, 23] root2Candidates0 34 seed else seed) = _
  rw [if_pos (by decide), root2Child0]
def root2Candidates1 : List Nat := [1217, 1973, 6719, 10433, 15641]
set_option maxRecDepth 8192 in
theorem root2Candidates1_checked : candidates edge 4 11 113 root2Tail2 seed.bound = root2Candidates1 := by
  unfold candidates
  have row : edge 113 = (SortedNatTree.contains FastRows.row113) := by
    change Data.row113.contains = (SortedNatTree.contains FastRows.row113)
    exact FastRows.row113_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child1 : search edge 3 [11, 113] root2Candidates1 124 seed = seed := by decide
theorem root2Step1 : scan edge (search edge 3) 4 [11] 11 root2Tail1 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail2 seed := by
  rw [root2Tail1, scan, if_neg (by decide), root2Candidates1_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail2
    (if 3 ≤ root2Candidates1.length then search edge 3 [11, 113] root2Candidates1 124 seed else seed) = _
  rw [if_pos (by decide), root2Child1]
def root2Candidates2 : List Nat := [1049, 1091, 1847, 2459, 3467, 5807, 7541, 10247, 10433, 11057, 13103, 21011]
set_option maxRecDepth 8192 in
theorem root2Candidates2_checked : candidates edge 4 11 239 root2Tail3 seed.bound = root2Candidates2 := by
  unfold candidates
  have row : edge 239 = (SortedNatTree.contains FastRows.row239) := by
    change Data.row239.contains = (SortedNatTree.contains FastRows.row239)
    exact FastRows.row239_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child2 : search edge 3 [11, 239] root2Candidates2 250 seed = seed := by decide
theorem root2Step2 : scan edge (search edge 3) 4 [11] 11 root2Tail2 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail3 seed := by
  rw [root2Tail2, scan, if_neg (by decide), root2Candidates2_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail3
    (if 3 ≤ root2Candidates2.length then search edge 3 [11, 239] root2Candidates2 250 seed else seed) = _
  rw [if_pos (by decide), root2Child2]
def root2Candidates3 : List Nat := [941, 7127, 8387, 12911, 15641, 15731, 21011, 22367]
set_option maxRecDepth 8192 in
theorem root2Candidates3_checked : candidates edge 4 11 251 root2Tail4 seed.bound = root2Candidates3 := by
  unfold candidates
  have row : edge 251 = (SortedNatTree.contains FastRows.row251) := by
    change Data.row251.contains = (SortedNatTree.contains FastRows.row251)
    exact FastRows.row251_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child3 : search edge 3 [11, 251] root2Candidates3 262 seed = seed := by decide
theorem root2Step3 : scan edge (search edge 3) 4 [11] 11 root2Tail3 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail4 seed := by
  rw [root2Tail3, scan, if_neg (by decide), root2Candidates3_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail4
    (if 3 ≤ root2Candidates3.length then search edge 3 [11, 251] root2Candidates3 262 seed else seed) = _
  rw [if_pos (by decide), root2Child3]
def root2Candidates4 : List Nat := [2213, 3779, 4493, 4967, 5849, 6791, 7517, 8219, 8831, 9839, 10859]
set_option maxRecDepth 8192 in
theorem root2Candidates4_checked : candidates edge 4 11 353 root2Tail5 seed.bound = root2Candidates4 := by
  unfold candidates
  have row : edge 353 = (SortedNatTree.contains FastRows.row353) := by
    change Data.row353.contains = (SortedNatTree.contains FastRows.row353)
    exact FastRows.row353_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child4 : search edge 3 [11, 353] root2Candidates4 364 seed = seed := by decide
theorem root2Step4 : scan edge (search edge 3) 4 [11] 11 root2Tail4 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail5 seed := by
  rw [root2Tail4, scan, if_neg (by decide), root2Candidates4_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail5
    (if 3 ≤ root2Candidates4.length then search edge 3 [11, 353] root2Candidates4 364 seed else seed) = _
  rw [if_pos (by decide), root2Child4]
def root2Candidates5 : List Nat := [4229, 5309]
set_option maxRecDepth 8192 in
theorem root2Candidates5_checked : candidates edge 4 11 503 root2Tail6 seed.bound = root2Candidates5 := by
  unfold candidates
  have row : edge 503 = (SortedNatTree.contains FastRows.row503) := by
    change Data.row503.contains = (SortedNatTree.contains FastRows.row503)
    exact FastRows.row503_checked.symm
  rw [row]
  decide
theorem root2Step5 : scan edge (search edge 3) 4 [11] 11 root2Tail5 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail6 seed := by
  rw [root2Tail5, scan, if_neg (by decide), root2Candidates5_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail6
    (if 3 ≤ root2Candidates5.length then search edge 3 [11, 503] root2Candidates5 514 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates6 : List Nat := [4229, 6911, 7127, 8681, 10859, 16937, 22397, 24593]
set_option maxRecDepth 8192 in
theorem root2Candidates6_checked : candidates edge 4 11 587 root2Tail7 seed.bound = root2Candidates6 := by
  unfold candidates
  have row : edge 587 = (SortedNatTree.contains FastRows.row587) := by
    change Data.row587.contains = (SortedNatTree.contains FastRows.row587)
    exact FastRows.row587_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child6 : search edge 3 [11, 587] root2Candidates6 598 seed = seed := by decide
theorem root2Step6 : scan edge (search edge 3) 4 [11] 11 root2Tail6 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail7 seed := by
  rw [root2Tail6, scan, if_neg (by decide), root2Candidates6_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail7
    (if 3 ≤ root2Candidates6.length then search edge 3 [11, 587] root2Candidates6 598 seed else seed) = _
  rw [if_pos (by decide), root2Child6]
def root2Candidates7 : List Nat := [7541, 7937, 8171, 8747, 11351]
set_option maxRecDepth 8192 in
theorem root2Candidates7_checked : candidates edge 4 11 701 root2Tail8 seed.bound = root2Candidates7 := by
  unfold candidates
  have row : edge 701 = (SortedNatTree.contains FastRows.row701) := by
    change Data.row701.contains = (SortedNatTree.contains FastRows.row701)
    exact FastRows.row701_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child7 : search edge 3 [11, 701] root2Candidates7 712 seed = seed := by decide
theorem root2Step7 : scan edge (search edge 3) 4 [11] 11 root2Tail7 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail8 seed := by
  rw [root2Tail7, scan, if_neg (by decide), root2Candidates7_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail8
    (if 3 ≤ root2Candidates7.length then search edge 3 [11, 701] root2Candidates7 712 seed else seed) = _
  rw [if_pos (by decide), root2Child7]
def root2Candidates8 : List Nat := [1871, 11057, 12569, 13751, 22133, 22397]
set_option maxRecDepth 8192 in
theorem root2Candidates8_checked : candidates edge 4 11 743 root2Tail9 seed.bound = root2Candidates8 := by
  unfold candidates
  have row : edge 743 = (SortedNatTree.contains FastRows.row743) := by
    change Data.row743.contains = (SortedNatTree.contains FastRows.row743)
    exact FastRows.row743_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child8 : search edge 3 [11, 743] root2Candidates8 754 seed = seed := by decide
theorem root2Step8 : scan edge (search edge 3) 4 [11] 11 root2Tail8 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail9 seed := by
  rw [root2Tail8, scan, if_neg (by decide), root2Candidates8_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail9
    (if 3 ≤ root2Candidates8.length then search edge 3 [11, 743] root2Candidates8 754 seed else seed) = _
  rw [if_pos (by decide), root2Child8]
def root2Candidates9 : List Nat := [1949, 7331, 7877, 8387, 13421, 17489, 21011]
set_option maxRecDepth 8192 in
theorem root2Candidates9_checked : candidates edge 4 11 839 root2Tail10 seed.bound = root2Candidates9 := by
  unfold candidates
  have row : edge 839 = (SortedNatTree.contains FastRows.row839) := by
    change Data.row839.contains = (SortedNatTree.contains FastRows.row839)
    exact FastRows.row839_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child9 : search edge 3 [11, 839] root2Candidates9 850 seed = seed := by decide
theorem root2Step9 : scan edge (search edge 3) 4 [11] 11 root2Tail9 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail10 seed := by
  rw [root2Tail9, scan, if_neg (by decide), root2Candidates9_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail10
    (if 3 ≤ root2Candidates9.length then search edge 3 [11, 839] root2Candidates9 850 seed else seed) = _
  rw [if_pos (by decide), root2Child9]
def root2Candidates10 : List Nat := [2297, 2621, 2927, 4259, 5849, 10247, 20369]
set_option maxRecDepth 8192 in
theorem root2Candidates10_checked : candidates edge 4 11 863 root2Tail11 seed.bound = root2Candidates10 := by
  unfold candidates
  have row : edge 863 = (SortedNatTree.contains FastRows.row863) := by
    change Data.row863.contains = (SortedNatTree.contains FastRows.row863)
    exact FastRows.row863_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child10 : search edge 3 [11, 863] root2Candidates10 874 seed = seed := by decide
theorem root2Step10 : scan edge (search edge 3) 4 [11] 11 root2Tail10 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail11 seed := by
  rw [root2Tail10, scan, if_neg (by decide), root2Candidates10_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail11
    (if 3 ≤ root2Candidates10.length then search edge 3 [11, 863] root2Candidates10 874 seed else seed) = _
  rw [if_pos (by decide), root2Child10]
def root2Candidates11 : List Nat := [1049, 6047, 8681, 12107, 15551, 23279]
set_option maxRecDepth 8192 in
theorem root2Candidates11_checked : candidates edge 4 11 941 root2Tail12 seed.bound = root2Candidates11 := by
  unfold candidates
  have row : edge 941 = (SortedNatTree.contains FastRows.row941) := by
    change Data.row941.contains = (SortedNatTree.contains FastRows.row941)
    exact FastRows.row941_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child11 : search edge 3 [11, 941] root2Candidates11 952 seed = seed := by decide
theorem root2Step11 : scan edge (search edge 3) 4 [11] 11 root2Tail11 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail12 seed := by
  rw [root2Tail11, scan, if_neg (by decide), root2Candidates11_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail12
    (if 3 ≤ root2Candidates11.length then search edge 3 [11, 941] root2Candidates11 952 seed else seed) = _
  rw [if_pos (by decide), root2Child11]
def root2Candidates12 : List Nat := [7109, 8423, 14591, 16937]
set_option maxRecDepth 8192 in
theorem root2Candidates12_checked : candidates edge 4 11 953 root2Tail13 seed.bound = root2Candidates12 := by
  unfold candidates
  have row : edge 953 = (SortedNatTree.contains FastRows.row953) := by
    change Data.row953.contains = (SortedNatTree.contains FastRows.row953)
    exact FastRows.row953_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child12 : search edge 3 [11, 953] root2Candidates12 964 seed = seed := by decide
theorem root2Step12 : scan edge (search edge 3) 4 [11] 11 root2Tail12 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail13 seed := by
  rw [root2Tail12, scan, if_neg (by decide), root2Candidates12_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail13
    (if 3 ≤ root2Candidates12.length then search edge 3 [11, 953] root2Candidates12 964 seed else seed) = _
  rw [if_pos (by decide), root2Child12]
def root2Candidates13 : List Nat := [1847, 2297, 4493, 7577, 8219, 11057, 12011, 20807]
set_option maxRecDepth 8192 in
theorem root2Candidates13_checked : candidates edge 4 11 1049 root2Tail14 seed.bound = root2Candidates13 := by
  unfold candidates
  have row : edge 1049 = (SortedNatTree.contains FastRows.row1049) := by
    change Data.row1049.contains = (SortedNatTree.contains FastRows.row1049)
    exact FastRows.row1049_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child13 : search edge 3 [11, 1049] root2Candidates13 1060 seed = seed := by decide
theorem root2Step13 : scan edge (search edge 3) 4 [11] 11 root2Tail13 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail14 seed := by
  rw [root2Tail13, scan, if_neg (by decide), root2Candidates13_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail14
    (if 3 ≤ root2Candidates13.length then search edge 3 [11, 1049] root2Candidates13 1060 seed else seed) = _
  rw [if_pos (by decide), root2Child13]
def root2Candidates14 : List Nat := [1847, 1949, 8081, 8681, 21557]
set_option maxRecDepth 8192 in
theorem root2Candidates14_checked : candidates edge 4 11 1091 root2Tail15 seed.bound = root2Candidates14 := by
  unfold candidates
  have row : edge 1091 = (SortedNatTree.contains FastRows.row1091) := by
    change Data.row1091.contains = (SortedNatTree.contains FastRows.row1091)
    exact FastRows.row1091_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child14 : search edge 3 [11, 1091] root2Candidates14 1102 seed = seed := by decide
theorem root2Step14 : scan edge (search edge 3) 4 [11] 11 root2Tail14 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail15 seed := by
  rw [root2Tail14, scan, if_neg (by decide), root2Candidates14_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail15
    (if 3 ≤ root2Candidates14.length then search edge 3 [11, 1091] root2Candidates14 1102 seed else seed) = _
  rw [if_pos (by decide), root2Child14]
def root2Candidates15 : List Nat := [2927, 5807, 5987, 6329, 7529, 15641, 23909]
set_option maxRecDepth 8192 in
theorem root2Candidates15_checked : candidates edge 4 11 1103 root2Tail16 seed.bound = root2Candidates15 := by
  unfold candidates
  have row : edge 1103 = (SortedNatTree.contains FastRows.row1103) := by
    change Data.row1103.contains = (SortedNatTree.contains FastRows.row1103)
    exact FastRows.row1103_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child15 : search edge 3 [11, 1103] root2Candidates15 1114 seed = seed := by decide
theorem root2Step15 : scan edge (search edge 3) 4 [11] 11 root2Tail15 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail16 seed := by
  rw [root2Tail15, scan, if_neg (by decide), root2Candidates15_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail16
    (if 3 ≤ root2Candidates15.length then search edge 3 [11, 1103] root2Candidates15 1114 seed else seed) = _
  rw [if_pos (by decide), root2Child15]
def root2Candidates16 : List Nat := [1427, 2843, 5153, 9689, 12653, 15551, 16937]
set_option maxRecDepth 8192 in
theorem root2Candidates16_checked : candidates edge 4 11 1217 root2Tail17 seed.bound = root2Candidates16 := by
  unfold candidates
  have row : edge 1217 = (SortedNatTree.contains FastRows.row1217) := by
    change Data.row1217.contains = (SortedNatTree.contains FastRows.row1217)
    exact FastRows.row1217_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child16 : search edge 3 [11, 1217] root2Candidates16 1228 seed = seed := by decide
theorem root2Step16 : scan edge (search edge 3) 4 [11] 11 root2Tail16 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail17 seed := by
  rw [root2Tail16, scan, if_neg (by decide), root2Candidates16_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail17
    (if 3 ≤ root2Candidates16.length then search edge 3 [11, 1217] root2Candidates16 1228 seed else seed) = _
  rw [if_pos (by decide), root2Child16]
def root2Candidates17 : List Nat := [2339, 7937, 8273, 22133, 22397]
set_option maxRecDepth 8192 in
theorem root2Candidates17_checked : candidates edge 4 11 1427 root2Tail18 seed.bound = root2Candidates17 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child17 : search edge 3 [11, 1427] root2Candidates17 1438 seed = seed := by decide
theorem root2Step17 : scan edge (search edge 3) 4 [11] 11 root2Tail17 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail18 seed := by
  rw [root2Tail17, scan, if_neg (by decide), root2Candidates17_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail18
    (if 3 ≤ root2Candidates17.length then search edge 3 [11, 1427] root2Candidates17 1438 seed else seed) = _
  rw [if_pos (by decide), root2Child17]
def root2Candidates18 : List Nat := [16943, 17489, 23873]
set_option maxRecDepth 8192 in
theorem root2Candidates18_checked : candidates edge 4 11 1487 root2Tail19 seed.bound = root2Candidates18 := by
  unfold candidates
  have row : edge 1487 = (SortedNatTree.contains FastRows.row1487) := by
    change Data.row1487.contains = (SortedNatTree.contains FastRows.row1487)
    exact FastRows.row1487_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child18 : search edge 3 [11, 1487] root2Candidates18 1498 seed = seed := by decide
theorem root2Step18 : scan edge (search edge 3) 4 [11] 11 root2Tail18 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail19 seed := by
  rw [root2Tail18, scan, if_neg (by decide), root2Candidates18_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail19
    (if 3 ≤ root2Candidates18.length then search edge 3 [11, 1487] root2Candidates18 1498 seed else seed) = _
  rw [if_pos (by decide), root2Child18]
def root2Candidates19 : List Nat := [3779, 7703, 24107]
set_option maxRecDepth 8192 in
theorem root2Candidates19_checked : candidates edge 4 11 1847 root2Tail20 seed.bound = root2Candidates19 := by
  unfold candidates
  have row : edge 1847 = (SortedNatTree.contains FastRows.row1847) := by
    change Data.row1847.contains = (SortedNatTree.contains FastRows.row1847)
    exact FastRows.row1847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child19 : search edge 3 [11, 1847] root2Candidates19 1858 seed = seed := by decide
theorem root2Step19 : scan edge (search edge 3) 4 [11] 11 root2Tail19 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail20 seed := by
  rw [root2Tail19, scan, if_neg (by decide), root2Candidates19_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail20
    (if 3 ≤ root2Candidates19.length then search edge 3 [11, 1847] root2Candidates19 1858 seed else seed) = _
  rw [if_pos (by decide), root2Child19]
def root2Candidates20 : List Nat := [8171, 8681, 19949]
set_option maxRecDepth 8192 in
theorem root2Candidates20_checked : candidates edge 4 11 1871 root2Tail21 seed.bound = root2Candidates20 := by
  unfold candidates
  have row : edge 1871 = (SortedNatTree.contains FastRows.row1871) := by
    change Data.row1871.contains = (SortedNatTree.contains FastRows.row1871)
    exact FastRows.row1871_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child20 : search edge 3 [11, 1871] root2Candidates20 1882 seed = seed := by decide
theorem root2Step20 : scan edge (search edge 3) 4 [11] 11 root2Tail20 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail21 seed := by
  rw [root2Tail20, scan, if_neg (by decide), root2Candidates20_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail21
    (if 3 ≤ root2Candidates20.length then search edge 3 [11, 1871] root2Candidates20 1882 seed else seed) = _
  rw [if_pos (by decide), root2Child20]
def root2Candidates21 : List Nat := [2621, 2657]
set_option maxRecDepth 8192 in
theorem root2Candidates21_checked : candidates edge 4 11 1949 root2Tail22 seed.bound = root2Candidates21 := by
  unfold candidates
  have row : edge 1949 = (SortedNatTree.contains FastRows.row1949) := by
    change Data.row1949.contains = (SortedNatTree.contains FastRows.row1949)
    exact FastRows.row1949_checked.symm
  rw [row]
  decide
theorem root2Step21 : scan edge (search edge 3) 4 [11] 11 root2Tail21 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail22 seed := by
  rw [root2Tail21, scan, if_neg (by decide), root2Candidates21_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail22
    (if 3 ≤ root2Candidates21.length then search edge 3 [11, 1949] root2Candidates21 1960 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates22 : List Nat := [4493, 10433, 11057, 14759]
set_option maxRecDepth 8192 in
theorem root2Candidates22_checked : candidates edge 4 11 1973 root2Tail23 seed.bound = root2Candidates22 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child22 : search edge 3 [11, 1973] root2Candidates22 1984 seed = seed := by decide
theorem root2Step22 : scan edge (search edge 3) 4 [11] 11 root2Tail22 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail23 seed := by
  rw [root2Tail22, scan, if_neg (by decide), root2Candidates22_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail23
    (if 3 ≤ root2Candidates22.length then search edge 3 [11, 1973] root2Candidates22 1984 seed else seed) = _
  rw [if_pos (by decide), root2Child22]
def root2Candidates23 : List Nat := [2297, 2339, 7757, 7877, 8219, 8747, 10433, 14759, 20807, 22739]
set_option maxRecDepth 8192 in
theorem root2Candidates23_checked : candidates edge 4 11 2069 root2Tail24 seed.bound = root2Candidates23 := by
  unfold candidates
  have row : edge 2069 = (SortedNatTree.contains FastRows.row2069) := by
    change Data.row2069.contains = (SortedNatTree.contains FastRows.row2069)
    exact FastRows.row2069_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child23 : search edge 3 [11, 2069] root2Candidates23 2080 seed = seed := by decide
theorem root2Step23 : scan edge (search edge 3) 4 [11] 11 root2Tail23 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail24 seed := by
  rw [root2Tail23, scan, if_neg (by decide), root2Candidates23_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail24
    (if 3 ≤ root2Candidates23.length then search edge 3 [11, 2069] root2Candidates23 2080 seed else seed) = _
  rw [if_pos (by decide), root2Child23]
def root2Candidates24 : List Nat := [9419, 22397]
set_option maxRecDepth 8192 in
theorem root2Candidates24_checked : candidates edge 4 11 2213 root2Tail25 seed.bound = root2Candidates24 := by
  unfold candidates
  have row : edge 2213 = (SortedNatTree.contains FastRows.row2213) := by
    change Data.row2213.contains = (SortedNatTree.contains FastRows.row2213)
    exact FastRows.row2213_checked.symm
  rw [row]
  decide
theorem root2Step24 : scan edge (search edge 3) 4 [11] 11 root2Tail24 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail25 seed := by
  rw [root2Tail24, scan, if_neg (by decide), root2Candidates24_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail25
    (if 3 ≤ root2Candidates24.length then search edge 3 [11, 2213] root2Candidates24 2224 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates25 : List Nat := [6791, 8423, 9419, 16937]
set_option maxRecDepth 8192 in
theorem root2Candidates25_checked : candidates edge 4 11 2237 root2Tail26 seed.bound = root2Candidates25 := by
  unfold candidates
  have row : edge 2237 = (SortedNatTree.contains FastRows.row2237) := by
    change Data.row2237.contains = (SortedNatTree.contains FastRows.row2237)
    exact FastRows.row2237_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child25 : search edge 3 [11, 2237] root2Candidates25 2248 seed = seed := by decide
theorem root2Step25 : scan edge (search edge 3) 4 [11] 11 root2Tail25 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail26 seed := by
  rw [root2Tail25, scan, if_neg (by decide), root2Candidates25_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail26
    (if 3 ≤ root2Candidates25.length then search edge 3 [11, 2237] root2Candidates25 2248 seed else seed) = _
  rw [if_pos (by decide), root2Child25]
def root2Candidates26 : List Nat := [4073, 4967, 7127, 7937, 9293, 12107]
set_option maxRecDepth 8192 in
theorem root2Candidates26_checked : candidates edge 4 11 2297 root2Tail27 seed.bound = root2Candidates26 := by
  unfold candidates
  have row : edge 2297 = (SortedNatTree.contains FastRows.row2297) := by
    change Data.row2297.contains = (SortedNatTree.contains FastRows.row2297)
    exact FastRows.row2297_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child26 : search edge 3 [11, 2297] root2Candidates26 2308 seed = seed := by decide
theorem root2Step26 : scan edge (search edge 3) 4 [11] 11 root2Tail26 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail27 seed := by
  rw [root2Tail26, scan, if_neg (by decide), root2Candidates26_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail27
    (if 3 ≤ root2Candidates26.length then search edge 3 [11, 2297] root2Candidates26 2308 seed else seed) = _
  rw [if_pos (by decide), root2Child26]
def root2Candidates27 : List Nat := [6047, 11351, 15641, 20807, 22679]
set_option maxRecDepth 8192 in
theorem root2Candidates27_checked : candidates edge 4 11 2339 root2Tail28 seed.bound = root2Candidates27 := by
  unfold candidates
  have row : edge 2339 = (SortedNatTree.contains FastRows.row2339) := by
    change Data.row2339.contains = (SortedNatTree.contains FastRows.row2339)
    exact FastRows.row2339_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child27 : search edge 3 [11, 2339] root2Candidates27 2350 seed = seed := by decide
theorem root2Step27 : scan edge (search edge 3) 4 [11] 11 root2Tail27 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail28 seed := by
  rw [root2Tail27, scan, if_neg (by decide), root2Candidates27_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail28
    (if 3 ≤ root2Candidates27.length then search edge 3 [11, 2339] root2Candidates27 2350 seed else seed) = _
  rw [if_pos (by decide), root2Child27]
def root2Candidates28 : List Nat := [2657, 2927, 8423, 10859, 11087, 21737]
set_option maxRecDepth 8192 in
theorem root2Candidates28_checked : candidates edge 4 11 2459 root2Tail29 seed.bound = root2Candidates28 := by
  unfold candidates
  have row : edge 2459 = (SortedNatTree.contains FastRows.row2459) := by
    change Data.row2459.contains = (SortedNatTree.contains FastRows.row2459)
    exact FastRows.row2459_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child28 : search edge 3 [11, 2459] root2Candidates28 2470 seed = seed := by decide
theorem root2Step28 : scan edge (search edge 3) 4 [11] 11 root2Tail28 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail29 seed := by
  rw [root2Tail28, scan, if_neg (by decide), root2Candidates28_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail29
    (if 3 ≤ root2Candidates28.length then search edge 3 [11, 2459] root2Candidates28 2470 seed else seed) = _
  rw [if_pos (by decide), root2Child28]
def root2Candidates29 : List Nat := [12569, 13421, 22397, 22739]
set_option maxRecDepth 8192 in
theorem root2Candidates29_checked : candidates edge 4 11 2621 root2Tail30 seed.bound = root2Candidates29 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child29 : search edge 3 [11, 2621] root2Candidates29 2632 seed = seed := by decide
theorem root2Step29 : scan edge (search edge 3) 4 [11] 11 root2Tail29 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail30 seed := by
  rw [root2Tail29, scan, if_neg (by decide), root2Candidates29_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail30
    (if 3 ≤ root2Candidates29.length then search edge 3 [11, 2621] root2Candidates29 2632 seed else seed) = _
  rw [if_pos (by decide), root2Child29]
def root2Candidates30 : List Nat := [5099, 12653]
set_option maxRecDepth 8192 in
theorem root2Candidates30_checked : candidates edge 4 11 2657 root2Tail31 seed.bound = root2Candidates30 := by
  unfold candidates
  have row : edge 2657 = (SortedNatTree.contains FastRows.row2657) := by
    change Data.row2657.contains = (SortedNatTree.contains FastRows.row2657)
    exact FastRows.row2657_checked.symm
  rw [row]
  decide
theorem root2Step30 : scan edge (search edge 3) 4 [11] 11 root2Tail30 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail31 seed := by
  rw [root2Tail30, scan, if_neg (by decide), root2Candidates30_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail31
    (if 3 ≤ root2Candidates30.length then search edge 3 [11, 2657] root2Candidates30 2668 seed else seed) = _
  rw [if_neg (by decide)]
def root2Candidates31 : List Nat := [5783, 5807, 6047, 8171, 8273, 8831, 12569, 12653, 16943, 22397]
set_option maxRecDepth 8192 in
theorem root2Candidates31_checked : candidates edge 4 11 2843 root2Tail32 seed.bound = root2Candidates31 := by
  unfold candidates
  have row : edge 2843 = (SortedNatTree.contains FastRows.row2843) := by
    change Data.row2843.contains = (SortedNatTree.contains FastRows.row2843)
    exact FastRows.row2843_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root2Child31 : search edge 3 [11, 2843] root2Candidates31 2854 seed = seed := by decide
theorem root2Step31 : scan edge (search edge 3) 4 [11] 11 root2Tail31 seed =
    scan edge (search edge 3) 4 [11] 11 root2Tail32 seed := by
  rw [root2Tail31, scan, if_neg (by decide), root2Candidates31_checked]
  change scan edge (search edge 3) 4 [11] 11 root2Tail32
    (if 3 ≤ root2Candidates31.length then search edge 3 [11, 2843] root2Candidates31 2854 seed else seed) = _
  rw [if_pos (by decide), root2Child31]
end ProjectEulerSolutions.PrimePairLarge.Search
