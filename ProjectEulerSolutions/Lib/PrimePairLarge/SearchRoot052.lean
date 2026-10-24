import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root52Candidates : List Nat := Data.row257
set_option maxRecDepth 8192 in
theorem root52Candidates_checked : candidates edge 5 0 257 tail53 winner.bound = root52Candidates := by
  apply candidates_eq_row edge 5 0 257 tail53 root52Candidates winner.bound tail53_ordered
  · decide
  · decide
  · funext q
    rfl
def root52Tail64 : List Nat := []
def root52Tail63 : List Nat := 25463 :: root52Tail64
def root52Tail62 : List Nat := 25307 :: root52Tail63
def root52Tail61 : List Nat := 25013 :: root52Tail62
def root52Tail60 : List Nat := 24683 :: root52Tail61
def root52Tail59 : List Nat := 23561 :: root52Tail60
def root52Tail58 : List Nat := 23447 :: root52Tail59
def root52Tail57 : List Nat := 23099 :: root52Tail58
def root52Tail56 : List Nat := 22871 :: root52Tail57
def root52Tail55 : List Nat := 22259 :: root52Tail56
def root52Tail54 : List Nat := 22079 :: root52Tail55
def root52Tail53 : List Nat := 20663 :: root52Tail54
def root52Tail52 : List Nat := 20441 :: root52Tail53
def root52Tail51 : List Nat := 20399 :: root52Tail52
def root52Tail50 : List Nat := 20369 :: root52Tail51
def root52Tail49 : List Nat := 19961 :: root52Tail50
def root52Tail48 : List Nat := 17987 :: root52Tail49
def root52Tail47 : List Nat := 15959 :: root52Tail48
def root52Tail46 : List Nat := 15791 :: root52Tail47
def root52Tail45 : List Nat := 15581 :: root52Tail46
def root52Tail44 : List Nat := 15401 :: root52Tail45
def root52Tail43 : List Nat := 14891 :: root52Tail44
def root52Tail42 : List Nat := 13451 :: root52Tail43
def root52Tail41 : List Nat := 13313 :: root52Tail42
def root52Tail40 : List Nat := 13217 :: root52Tail41
def root52Tail39 : List Nat := 13007 :: root52Tail40
def root52Tail38 : List Nat := 12923 :: root52Tail39
def root52Tail37 : List Nat := 12539 :: root52Tail38
def root52Tail36 : List Nat := 11321 :: root52Tail37
def root52Tail35 : List Nat := 10859 :: root52Tail36
def root52Tail34 : List Nat := 9221 :: root52Tail35
def root52Tail33 : List Nat := 8963 :: root52Tail34
def root52Tail32 : List Nat := 8423 :: root52Tail33
def root52Tail31 : List Nat := 8273 :: root52Tail32
def root52Tail30 : List Nat := 7907 :: root52Tail31
def root52Tail29 : List Nat := 7283 :: root52Tail30
def root52Tail28 : List Nat := 7079 :: root52Tail29
def root52Tail27 : List Nat := 6983 :: root52Tail28
def root52Tail26 : List Nat := 5939 :: root52Tail27
def root52Tail25 : List Nat := 5351 :: root52Tail26
def root52Tail24 : List Nat := 5021 :: root52Tail25
def root52Tail23 : List Nat := 4967 :: root52Tail24
def root52Tail22 : List Nat := 4889 :: root52Tail23
def root52Tail21 : List Nat := 4409 :: root52Tail22
def root52Tail20 : List Nat := 4337 :: root52Tail21
def root52Tail19 : List Nat := 3833 :: root52Tail20
def root52Tail18 : List Nat := 3413 :: root52Tail19
def root52Tail17 : List Nat := 3209 :: root52Tail18
def root52Tail16 : List Nat := 3023 :: root52Tail17
def root52Tail15 : List Nat := 2939 :: root52Tail16
def root52Tail14 : List Nat := 2741 :: root52Tail15
def root52Tail13 : List Nat := 2621 :: root52Tail14
def root52Tail12 : List Nat := 2393 :: root52Tail13
def root52Tail11 : List Nat := 2267 :: root52Tail12
def root52Tail10 : List Nat := 2243 :: root52Tail11
def root52Tail9 : List Nat := 2027 :: root52Tail10
def root52Tail8 : List Nat := 1949 :: root52Tail9
def root52Tail7 : List Nat := 1823 :: root52Tail8
def root52Tail6 : List Nat := 1721 :: root52Tail7
def root52Tail5 : List Nat := 1697 :: root52Tail6
def root52Tail4 : List Nat := 1619 :: root52Tail5
def root52Tail3 : List Nat := 1451 :: root52Tail4
def root52Tail2 : List Nat := 1277 :: root52Tail3
def root52Tail1 : List Nat := 293 :: root52Tail2
def root52Tail0 : List Nat := 263 :: root52Tail1
def root52Candidates0 : List Nat := [293, 11321, 15581]
set_option maxRecDepth 8192 in
theorem root52Candidates0_checked : candidates edge 4 257 263 root52Tail1 winner.bound = root52Candidates0 := by
  unfold candidates
  have row : edge 263 = (SortedNatTree.contains FastRows.row263) := by
    change Data.row263.contains = (SortedNatTree.contains FastRows.row263)
    exact FastRows.row263_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root52Child0 : search edge 3 [257, 263] root52Candidates0 520 winner = winner := by decide
theorem root52Step0 : scan edge (search edge 3) 4 [257] 257 root52Tail0 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail1 winner := by
  rw [root52Tail0, scan, if_neg (by decide), root52Candidates0_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail1
    (if 3 ≤ root52Candidates0.length then search edge 3 [257, 263] root52Candidates0 520 winner else winner) = _
  rw [if_pos (by decide), root52Child0]
def root52Candidates1 : List Nat := [1823, 12923, 15401, 23447]
set_option maxRecDepth 8192 in
theorem root52Candidates1_checked : candidates edge 4 257 293 root52Tail2 winner.bound = root52Candidates1 := by
  unfold candidates
  have row : edge 293 = (SortedNatTree.contains FastRows.row293) := by
    change Data.row293.contains = (SortedNatTree.contains FastRows.row293)
    exact FastRows.row293_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root52Child1 : search edge 3 [257, 293] root52Candidates1 550 winner = winner := by decide
theorem root52Step1 : scan edge (search edge 3) 4 [257] 257 root52Tail1 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail2 winner := by
  rw [root52Tail1, scan, if_neg (by decide), root52Candidates1_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail2
    (if 3 ≤ root52Candidates1.length then search edge 3 [257, 293] root52Candidates1 550 winner else winner) = _
  rw [if_pos (by decide), root52Child1]
def root52Candidates2 : List Nat := [4889, 4967]
set_option maxRecDepth 8192 in
theorem root52Candidates2_checked : candidates edge 4 257 1277 root52Tail3 winner.bound = root52Candidates2 := by
  unfold candidates
  have row : edge 1277 = (SortedNatTree.contains FastRows.row1277) := by
    change Data.row1277.contains = (SortedNatTree.contains FastRows.row1277)
    exact FastRows.row1277_checked.symm
  rw [row]
  decide
theorem root52Step2 : scan edge (search edge 3) 4 [257] 257 root52Tail2 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail3 winner := by
  rw [root52Tail2, scan, if_neg (by decide), root52Candidates2_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail3
    (if 3 ≤ root52Candidates2.length then search edge 3 [257, 1277] root52Candidates2 1534 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates3 : List Nat := [3413, 7907, 15959]
set_option maxRecDepth 8192 in
theorem root52Candidates3_checked : candidates edge 4 257 1451 root52Tail4 winner.bound = root52Candidates3 := by
  unfold candidates
  have row : edge 1451 = (SortedNatTree.contains FastRows.row1451) := by
    change Data.row1451.contains = (SortedNatTree.contains FastRows.row1451)
    exact FastRows.row1451_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root52Child3 : search edge 3 [257, 1451] root52Candidates3 1708 winner = winner := by decide
theorem root52Step3 : scan edge (search edge 3) 4 [257] 257 root52Tail3 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail4 winner := by
  rw [root52Tail3, scan, if_neg (by decide), root52Candidates3_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail4
    (if 3 ≤ root52Candidates3.length then search edge 3 [257, 1451] root52Candidates3 1708 winner else winner) = _
  rw [if_pos (by decide), root52Child3]
def root52Candidates4 : List Nat := []
set_option maxRecDepth 8192 in
theorem root52Candidates4_checked : candidates edge 4 257 1619 root52Tail5 winner.bound = root52Candidates4 := by
  unfold candidates
  have row : edge 1619 = (SortedNatTree.contains FastRows.row1619) := by
    change Data.row1619.contains = (SortedNatTree.contains FastRows.row1619)
    exact FastRows.row1619_checked.symm
  rw [row]
  decide
theorem root52Step4 : scan edge (search edge 3) 4 [257] 257 root52Tail4 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail5 winner := by
  rw [root52Tail4, scan, if_neg (by decide), root52Candidates4_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail5
    (if 3 ≤ root52Candidates4.length then search edge 3 [257, 1619] root52Candidates4 1876 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates5 : List Nat := [2027, 2741]
set_option maxRecDepth 8192 in
theorem root52Candidates5_checked : candidates edge 4 257 1697 root52Tail6 winner.bound = root52Candidates5 := by
  unfold candidates
  have row : edge 1697 = (SortedNatTree.contains FastRows.row1697) := by
    change Data.row1697.contains = (SortedNatTree.contains FastRows.row1697)
    exact FastRows.row1697_checked.symm
  rw [row]
  decide
theorem root52Step5 : scan edge (search edge 3) 4 [257] 257 root52Tail5 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail6 winner := by
  rw [root52Tail5, scan, if_neg (by decide), root52Candidates5_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail6
    (if 3 ≤ root52Candidates5.length then search edge 3 [257, 1697] root52Candidates5 1954 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates6 : List Nat := [2621, 7283, 8273]
set_option maxRecDepth 8192 in
theorem root52Candidates6_checked : candidates edge 4 257 1721 root52Tail7 winner.bound = root52Candidates6 := by
  unfold candidates
  have row : edge 1721 = (SortedNatTree.contains FastRows.row1721) := by
    change Data.row1721.contains = (SortedNatTree.contains FastRows.row1721)
    exact FastRows.row1721_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root52Child6 : search edge 3 [257, 1721] root52Candidates6 1978 winner = winner := by decide
theorem root52Step6 : scan edge (search edge 3) 4 [257] 257 root52Tail6 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail7 winner := by
  rw [root52Tail6, scan, if_neg (by decide), root52Candidates6_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail7
    (if 3 ≤ root52Candidates6.length then search edge 3 [257, 1721] root52Candidates6 1978 winner else winner) = _
  rw [if_pos (by decide), root52Child6]
def root52Candidates7 : List Nat := [3023, 6983, 10859]
set_option maxRecDepth 8192 in
theorem root52Candidates7_checked : candidates edge 4 257 1823 root52Tail8 winner.bound = root52Candidates7 := by
  unfold candidates
  have row : edge 1823 = (SortedNatTree.contains FastRows.row1823) := by
    change Data.row1823.contains = (SortedNatTree.contains FastRows.row1823)
    exact FastRows.row1823_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root52Child7 : search edge 3 [257, 1823] root52Candidates7 2080 winner = winner := by decide
theorem root52Step7 : scan edge (search edge 3) 4 [257] 257 root52Tail7 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail8 winner := by
  rw [root52Tail7, scan, if_neg (by decide), root52Candidates7_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail8
    (if 3 ≤ root52Candidates7.length then search edge 3 [257, 1823] root52Candidates7 2080 winner else winner) = _
  rw [if_pos (by decide), root52Child7]
def root52Candidates8 : List Nat := [2027, 2621, 4337, 19961]
set_option maxRecDepth 8192 in
theorem root52Candidates8_checked : candidates edge 4 257 1949 root52Tail9 winner.bound = root52Candidates8 := by
  unfold candidates
  have row : edge 1949 = (SortedNatTree.contains FastRows.row1949) := by
    change Data.row1949.contains = (SortedNatTree.contains FastRows.row1949)
    exact FastRows.row1949_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root52Child8 : search edge 3 [257, 1949] root52Candidates8 2206 winner = winner := by decide
theorem root52Step8 : scan edge (search edge 3) 4 [257] 257 root52Tail8 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail9 winner := by
  rw [root52Tail8, scan, if_neg (by decide), root52Candidates8_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail9
    (if 3 ≤ root52Candidates8.length then search edge 3 [257, 1949] root52Candidates8 2206 winner else winner) = _
  rw [if_pos (by decide), root52Child8]
def root52Candidates9 : List Nat := []
set_option maxRecDepth 8192 in
theorem root52Candidates9_checked : candidates edge 4 257 2027 root52Tail10 winner.bound = root52Candidates9 := by
  unfold candidates
  have row : edge 2027 = (SortedNatTree.contains FastRows.row2027) := by
    change Data.row2027.contains = (SortedNatTree.contains FastRows.row2027)
    exact FastRows.row2027_checked.symm
  rw [row]
  decide
theorem root52Step9 : scan edge (search edge 3) 4 [257] 257 root52Tail9 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail10 winner := by
  rw [root52Tail9, scan, if_neg (by decide), root52Candidates9_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail10
    (if 3 ≤ root52Candidates9.length then search edge 3 [257, 2027] root52Candidates9 2284 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates10 : List Nat := []
set_option maxRecDepth 8192 in
theorem root52Candidates10_checked : candidates edge 4 257 2243 root52Tail11 winner.bound = root52Candidates10 := by
  unfold candidates
  have row : edge 2243 = (SortedNatTree.contains FastRows.row2243) := by
    change Data.row2243.contains = (SortedNatTree.contains FastRows.row2243)
    exact FastRows.row2243_checked.symm
  rw [row]
  decide
theorem root52Step10 : scan edge (search edge 3) 4 [257] 257 root52Tail10 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail11 winner := by
  rw [root52Tail10, scan, if_neg (by decide), root52Candidates10_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail11
    (if 3 ≤ root52Candidates10.length then search edge 3 [257, 2243] root52Candidates10 2500 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates11 : List Nat := [2939]
set_option maxRecDepth 8192 in
theorem root52Candidates11_checked : candidates edge 4 257 2267 root52Tail12 winner.bound = root52Candidates11 := by
  unfold candidates
  have row : edge 2267 = (SortedNatTree.contains FastRows.row2267) := by
    change Data.row2267.contains = (SortedNatTree.contains FastRows.row2267)
    exact FastRows.row2267_checked.symm
  rw [row]
  decide
theorem root52Step11 : scan edge (search edge 3) 4 [257] 257 root52Tail11 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail12 winner := by
  rw [root52Tail11, scan, if_neg (by decide), root52Candidates11_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail12
    (if 3 ≤ root52Candidates11.length then search edge 3 [257, 2267] root52Candidates11 2524 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates12 : List Nat := [19961, 22079]
set_option maxRecDepth 8192 in
theorem root52Candidates12_checked : candidates edge 4 257 2393 root52Tail13 winner.bound = root52Candidates12 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
theorem root52Step12 : scan edge (search edge 3) 4 [257] 257 root52Tail12 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail13 winner := by
  rw [root52Tail12, scan, if_neg (by decide), root52Candidates12_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail13
    (if 3 ≤ root52Candidates12.length then search edge 3 [257, 2393] root52Candidates12 2650 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates13 : List Nat := [17987]
set_option maxRecDepth 8192 in
theorem root52Candidates13_checked : candidates edge 4 257 2621 root52Tail14 winner.bound = root52Candidates13 := by
  unfold candidates
  have row : edge 2621 = (SortedNatTree.contains FastRows.row2621) := by
    change Data.row2621.contains = (SortedNatTree.contains FastRows.row2621)
    exact FastRows.row2621_checked.symm
  rw [row]
  decide
theorem root52Step13 : scan edge (search edge 3) 4 [257] 257 root52Tail13 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail14 winner := by
  rw [root52Tail13, scan, if_neg (by decide), root52Candidates13_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail14
    (if 3 ≤ root52Candidates13.length then search edge 3 [257, 2621] root52Candidates13 2878 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates14 : List Nat := []
set_option maxRecDepth 8192 in
theorem root52Candidates14_checked : candidates edge 4 257 2741 root52Tail15 winner.bound = root52Candidates14 := by
  unfold candidates
  have row : edge 2741 = (SortedNatTree.contains FastRows.row2741) := by
    change Data.row2741.contains = (SortedNatTree.contains FastRows.row2741)
    exact FastRows.row2741_checked.symm
  rw [row]
  decide
theorem root52Step14 : scan edge (search edge 3) 4 [257] 257 root52Tail14 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail15 winner := by
  rw [root52Tail14, scan, if_neg (by decide), root52Candidates14_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail15
    (if 3 ≤ root52Candidates14.length then search edge 3 [257, 2741] root52Candidates14 2998 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates15 : List Nat := [20441]
set_option maxRecDepth 8192 in
theorem root52Candidates15_checked : candidates edge 4 257 2939 root52Tail16 winner.bound = root52Candidates15 := by
  unfold candidates
  have row : edge 2939 = (SortedNatTree.contains FastRows.row2939) := by
    change Data.row2939.contains = (SortedNatTree.contains FastRows.row2939)
    exact FastRows.row2939_checked.symm
  rw [row]
  decide
theorem root52Step15 : scan edge (search edge 3) 4 [257] 257 root52Tail15 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail16 winner := by
  rw [root52Tail15, scan, if_neg (by decide), root52Candidates15_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail16
    (if 3 ≤ root52Candidates15.length then search edge 3 [257, 2939] root52Candidates15 3196 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates16 : List Nat := [19961]
set_option maxRecDepth 8192 in
theorem root52Candidates16_checked : candidates edge 4 257 3023 root52Tail17 winner.bound = root52Candidates16 := by
  unfold candidates
  have row : edge 3023 = (SortedNatTree.contains FastRows.row3023) := by
    change Data.row3023.contains = (SortedNatTree.contains FastRows.row3023)
    exact FastRows.row3023_checked.symm
  rw [row]
  decide
theorem root52Step16 : scan edge (search edge 3) 4 [257] 257 root52Tail16 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail17 winner := by
  rw [root52Tail16, scan, if_neg (by decide), root52Candidates16_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail17
    (if 3 ≤ root52Candidates16.length then search edge 3 [257, 3023] root52Candidates16 3280 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates17 : List Nat := [3833, 12539]
set_option maxRecDepth 8192 in
theorem root52Candidates17_checked : candidates edge 4 257 3209 root52Tail18 winner.bound = root52Candidates17 := by
  unfold candidates
  have row : edge 3209 = (SortedNatTree.contains FastRows.row3209) := by
    change Data.row3209.contains = (SortedNatTree.contains FastRows.row3209)
    exact FastRows.row3209_checked.symm
  rw [row]
  decide
theorem root52Step17 : scan edge (search edge 3) 4 [257] 257 root52Tail17 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail18 winner := by
  rw [root52Tail17, scan, if_neg (by decide), root52Candidates17_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail18
    (if 3 ≤ root52Candidates17.length then search edge 3 [257, 3209] root52Candidates17 3466 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates18 : List Nat := [5939]
set_option maxRecDepth 8192 in
theorem root52Candidates18_checked : candidates edge 4 257 3413 root52Tail19 winner.bound = root52Candidates18 := by
  unfold candidates
  have row : edge 3413 = (SortedNatTree.contains FastRows.row3413) := by
    change Data.row3413.contains = (SortedNatTree.contains FastRows.row3413)
    exact FastRows.row3413_checked.symm
  rw [row]
  decide
theorem root52Step18 : scan edge (search edge 3) 4 [257] 257 root52Tail18 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail19 winner := by
  rw [root52Tail18, scan, if_neg (by decide), root52Candidates18_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail19
    (if 3 ≤ root52Candidates18.length then search edge 3 [257, 3413] root52Candidates18 3670 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates19 : List Nat := [4889, 6983, 14891]
set_option maxRecDepth 8192 in
theorem root52Candidates19_checked : candidates edge 4 257 3833 root52Tail20 winner.bound = root52Candidates19 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root52Child19 : search edge 3 [257, 3833] root52Candidates19 4090 winner = winner := by decide
theorem root52Step19 : scan edge (search edge 3) 4 [257] 257 root52Tail19 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail20 winner := by
  rw [root52Tail19, scan, if_neg (by decide), root52Candidates19_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail20
    (if 3 ≤ root52Candidates19.length then search edge 3 [257, 3833] root52Candidates19 4090 winner else winner) = _
  rw [if_pos (by decide), root52Child19]
def root52Candidates20 : List Nat := [4889, 11321]
set_option maxRecDepth 8192 in
theorem root52Candidates20_checked : candidates edge 4 257 4337 root52Tail21 winner.bound = root52Candidates20 := by
  unfold candidates
  have row : edge 4337 = (SortedNatTree.contains FastRows.row4337) := by
    change Data.row4337.contains = (SortedNatTree.contains FastRows.row4337)
    exact FastRows.row4337_checked.symm
  rw [row]
  decide
theorem root52Step20 : scan edge (search edge 3) 4 [257] 257 root52Tail20 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail21 winner := by
  rw [root52Tail20, scan, if_neg (by decide), root52Candidates20_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail21
    (if 3 ≤ root52Candidates20.length then search edge 3 [257, 4337] root52Candidates20 4594 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root52Candidates21_checked : candidates edge 4 257 4409 root52Tail22 winner.bound = root52Candidates21 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root52Step21 : scan edge (search edge 3) 4 [257] 257 root52Tail21 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail22 winner := by
  rw [root52Tail21, scan, if_neg (by decide), root52Candidates21_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail22
    (if 3 ≤ root52Candidates21.length then search edge 3 [257, 4409] root52Candidates21 4666 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates22 : List Nat := [17987]
set_option maxRecDepth 8192 in
theorem root52Candidates22_checked : candidates edge 4 257 4889 root52Tail23 winner.bound = root52Candidates22 := by
  unfold candidates
  have row : edge 4889 = (SortedNatTree.contains FastRows.row4889) := by
    change Data.row4889.contains = (SortedNatTree.contains FastRows.row4889)
    exact FastRows.row4889_checked.symm
  rw [row]
  decide
theorem root52Step22 : scan edge (search edge 3) 4 [257] 257 root52Tail22 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail23 winner := by
  rw [root52Tail22, scan, if_neg (by decide), root52Candidates22_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail23
    (if 3 ≤ root52Candidates22.length then search edge 3 [257, 4889] root52Candidates22 5146 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates23 : List Nat := [5939, 12539]
set_option maxRecDepth 8192 in
theorem root52Candidates23_checked : candidates edge 4 257 4967 root52Tail24 winner.bound = root52Candidates23 := by
  unfold candidates
  have row : edge 4967 = (SortedNatTree.contains FastRows.row4967) := by
    change Data.row4967.contains = (SortedNatTree.contains FastRows.row4967)
    exact FastRows.row4967_checked.symm
  rw [row]
  decide
theorem root52Step23 : scan edge (search edge 3) 4 [257] 257 root52Tail23 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail24 winner := by
  rw [root52Tail23, scan, if_neg (by decide), root52Candidates23_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail24
    (if 3 ≤ root52Candidates23.length then search edge 3 [257, 4967] root52Candidates23 5224 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates24 : List Nat := []
set_option maxRecDepth 8192 in
theorem root52Candidates24_checked : candidates edge 4 257 5021 root52Tail25 winner.bound = root52Candidates24 := by
  unfold candidates
  have row : edge 5021 = (SortedNatTree.contains FastRows.row5021) := by
    change Data.row5021.contains = (SortedNatTree.contains FastRows.row5021)
    exact FastRows.row5021_checked.symm
  rw [row]
  decide
theorem root52Step24 : scan edge (search edge 3) 4 [257] 257 root52Tail24 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail25 winner := by
  rw [root52Tail24, scan, if_neg (by decide), root52Candidates24_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail25
    (if 3 ≤ root52Candidates24.length then search edge 3 [257, 5021] root52Candidates24 5278 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates25 : List Nat := [11321, 13451]
set_option maxRecDepth 8192 in
theorem root52Candidates25_checked : candidates edge 4 257 5351 root52Tail26 winner.bound = root52Candidates25 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root52Step25 : scan edge (search edge 3) 4 [257] 257 root52Tail25 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail26 winner := by
  rw [root52Tail25, scan, if_neg (by decide), root52Candidates25_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail26
    (if 3 ≤ root52Candidates25.length then search edge 3 [257, 5351] root52Candidates25 5608 winner else winner) = _
  rw [if_neg (by decide)]
def root52Candidates26 : List Nat := []
set_option maxRecDepth 8192 in
theorem root52Candidates26_checked : candidates edge 4 257 5939 root52Tail27 winner.bound = root52Candidates26 := by
  unfold candidates
  have row : edge 5939 = (SortedNatTree.contains FastRows.row5939) := by
    change Data.row5939.contains = (SortedNatTree.contains FastRows.row5939)
    exact FastRows.row5939_checked.symm
  rw [row]
  decide
theorem root52Step26 : scan edge (search edge 3) 4 [257] 257 root52Tail26 winner =
    scan edge (search edge 3) 4 [257] 257 root52Tail27 winner := by
  rw [root52Tail26, scan, if_neg (by decide), root52Candidates26_checked]
  change scan edge (search edge 3) 4 [257] 257 root52Tail27
    (if 3 ≤ root52Candidates26.length then search edge 3 [257, 5939] root52Candidates26 6196 winner else winner) = _
  rw [if_neg (by decide)]
theorem root52Scan27 : scan edge (search edge 3) 4 [257] 257 root52Tail27 winner = winner := by rw [root52Tail27, scan, if_pos (by decide)]
theorem root52Scan26 : scan edge (search edge 3) 4 [257] 257 root52Tail26 winner = winner :=
  (root52Step26).trans root52Scan27
theorem root52Scan25 : scan edge (search edge 3) 4 [257] 257 root52Tail25 winner = winner :=
  (root52Step25).trans root52Scan26
theorem root52Scan24 : scan edge (search edge 3) 4 [257] 257 root52Tail24 winner = winner :=
  (root52Step24).trans root52Scan25
theorem root52Scan23 : scan edge (search edge 3) 4 [257] 257 root52Tail23 winner = winner :=
  (root52Step23).trans root52Scan24
theorem root52Scan22 : scan edge (search edge 3) 4 [257] 257 root52Tail22 winner = winner :=
  (root52Step22).trans root52Scan23
theorem root52Scan21 : scan edge (search edge 3) 4 [257] 257 root52Tail21 winner = winner :=
  (root52Step21).trans root52Scan22
theorem root52Scan20 : scan edge (search edge 3) 4 [257] 257 root52Tail20 winner = winner :=
  (root52Step20).trans root52Scan21
theorem root52Scan19 : scan edge (search edge 3) 4 [257] 257 root52Tail19 winner = winner :=
  (root52Step19).trans root52Scan20
theorem root52Scan18 : scan edge (search edge 3) 4 [257] 257 root52Tail18 winner = winner :=
  (root52Step18).trans root52Scan19
theorem root52Scan17 : scan edge (search edge 3) 4 [257] 257 root52Tail17 winner = winner :=
  (root52Step17).trans root52Scan18
theorem root52Scan16 : scan edge (search edge 3) 4 [257] 257 root52Tail16 winner = winner :=
  (root52Step16).trans root52Scan17
theorem root52Scan15 : scan edge (search edge 3) 4 [257] 257 root52Tail15 winner = winner :=
  (root52Step15).trans root52Scan16
theorem root52Scan14 : scan edge (search edge 3) 4 [257] 257 root52Tail14 winner = winner :=
  (root52Step14).trans root52Scan15
theorem root52Scan13 : scan edge (search edge 3) 4 [257] 257 root52Tail13 winner = winner :=
  (root52Step13).trans root52Scan14
theorem root52Scan12 : scan edge (search edge 3) 4 [257] 257 root52Tail12 winner = winner :=
  (root52Step12).trans root52Scan13
theorem root52Scan11 : scan edge (search edge 3) 4 [257] 257 root52Tail11 winner = winner :=
  (root52Step11).trans root52Scan12
theorem root52Scan10 : scan edge (search edge 3) 4 [257] 257 root52Tail10 winner = winner :=
  (root52Step10).trans root52Scan11
theorem root52Scan9 : scan edge (search edge 3) 4 [257] 257 root52Tail9 winner = winner :=
  (root52Step9).trans root52Scan10
theorem root52Scan8 : scan edge (search edge 3) 4 [257] 257 root52Tail8 winner = winner :=
  (root52Step8).trans root52Scan9
theorem root52Scan7 : scan edge (search edge 3) 4 [257] 257 root52Tail7 winner = winner :=
  (root52Step7).trans root52Scan8
theorem root52Scan6 : scan edge (search edge 3) 4 [257] 257 root52Tail6 winner = winner :=
  (root52Step6).trans root52Scan7
theorem root52Scan5 : scan edge (search edge 3) 4 [257] 257 root52Tail5 winner = winner :=
  (root52Step5).trans root52Scan6
theorem root52Scan4 : scan edge (search edge 3) 4 [257] 257 root52Tail4 winner = winner :=
  (root52Step4).trans root52Scan5
theorem root52Scan3 : scan edge (search edge 3) 4 [257] 257 root52Tail3 winner = winner :=
  (root52Step3).trans root52Scan4
theorem root52Scan2 : scan edge (search edge 3) 4 [257] 257 root52Tail2 winner = winner :=
  (root52Step2).trans root52Scan3
theorem root52Scan1 : scan edge (search edge 3) 4 [257] 257 root52Tail1 winner = winner :=
  (root52Step1).trans root52Scan2
theorem root52Scan0 : scan edge (search edge 3) 4 [257] 257 root52Tail0 winner = winner :=
  (root52Step0).trans root52Scan1
set_option maxRecDepth 4096 in
theorem root52Branch : search edge 4 [257] root52Candidates 257 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [257] 257 root52Tail0 winner = winner
  exact root52Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
