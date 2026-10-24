import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root74Candidates : List Nat := Data.row389
set_option maxRecDepth 8192 in
theorem root74Candidates_checked : candidates edge 5 0 389 tail75 winner.bound = root74Candidates := by
  apply candidates_eq_row edge 5 0 389 tail75 root74Candidates winner.bound tail75_ordered
  · decide
  · decide
  · funext q
    rfl
def root74Tail83 : List Nat := []
def root74Tail82 : List Nat := 25643 :: root74Tail83
def root74Tail81 : List Nat := 25391 :: root74Tail82
def root74Tail80 : List Nat := 24953 :: root74Tail81
def root74Tail79 : List Nat := 24107 :: root74Tail80
def root74Tail78 : List Nat := 24071 :: root74Tail79
def root74Tail77 : List Nat := 23747 :: root74Tail78
def root74Tail76 : List Nat := 23627 :: root74Tail77
def root74Tail75 : List Nat := 23603 :: root74Tail76
def root74Tail74 : List Nat := 23333 :: root74Tail75
def root74Tail73 : List Nat := 23117 :: root74Tail74
def root74Tail72 : List Nat := 22787 :: root74Tail73
def root74Tail71 : List Nat := 21401 :: root74Tail72
def root74Tail70 : List Nat := 21383 :: root74Tail71
def root74Tail69 : List Nat := 20177 :: root74Tail70
def root74Tail68 : List Nat := 19571 :: root74Tail69
def root74Tail67 : List Nat := 19469 :: root74Tail68
def root74Tail66 : List Nat := 19319 :: root74Tail67
def root74Tail65 : List Nat := 19157 :: root74Tail66
def root74Tail64 : List Nat := 18839 :: root74Tail65
def root74Tail63 : List Nat := 18587 :: root74Tail64
def root74Tail62 : List Nat := 18401 :: root74Tail63
def root74Tail61 : List Nat := 18233 :: root74Tail62
def root74Tail60 : List Nat := 17891 :: root74Tail61
def root74Tail59 : List Nat := 17387 :: root74Tail60
def root74Tail58 : List Nat := 17093 :: root74Tail59
def root74Tail57 : List Nat := 16091 :: root74Tail58
def root74Tail56 : List Nat := 15647 :: root74Tail57
def root74Tail55 : List Nat := 15269 :: root74Tail56
def root74Tail54 : List Nat := 15263 :: root74Tail55
def root74Tail53 : List Nat := 14891 :: root74Tail54
def root74Tail52 : List Nat := 14867 :: root74Tail53
def root74Tail51 : List Nat := 14537 :: root74Tail52
def root74Tail50 : List Nat := 14153 :: root74Tail51
def root74Tail49 : List Nat := 14009 :: root74Tail50
def root74Tail48 : List Nat := 13229 :: root74Tail49
def root74Tail47 : List Nat := 13001 :: root74Tail48
def root74Tail46 : List Nat := 12953 :: root74Tail47
def root74Tail45 : List Nat := 12743 :: root74Tail46
def root74Tail44 : List Nat := 11783 :: root74Tail45
def root74Tail43 : List Nat := 11597 :: root74Tail44
def root74Tail42 : List Nat := 11471 :: root74Tail43
def root74Tail41 : List Nat := 11399 :: root74Tail42
def root74Tail40 : List Nat := 11321 :: root74Tail41
def root74Tail39 : List Nat := 11027 :: root74Tail40
def root74Tail38 : List Nat := 10211 :: root74Tail39
def root74Tail37 : List Nat := 10163 :: root74Tail38
def root74Tail36 : List Nat := 9227 :: root74Tail37
def root74Tail35 : List Nat := 8513 :: root74Tail36
def root74Tail34 : List Nat := 8123 :: root74Tail35
def root74Tail33 : List Nat := 8087 :: root74Tail34
def root74Tail32 : List Nat := 7937 :: root74Tail33
def root74Tail31 : List Nat := 7853 :: root74Tail32
def root74Tail30 : List Nat := 7727 :: root74Tail31
def root74Tail29 : List Nat := 7703 :: root74Tail30
def root74Tail28 : List Nat := 7499 :: root74Tail29
def root74Tail27 : List Nat := 6947 :: root74Tail28
def root74Tail26 : List Nat := 6257 :: root74Tail27
def root74Tail25 : List Nat := 6089 :: root74Tail26
def root74Tail24 : List Nat := 6029 :: root74Tail25
def root74Tail23 : List Nat := 5351 :: root74Tail24
def root74Tail22 : List Nat := 5231 :: root74Tail23
def root74Tail21 : List Nat := 4733 :: root74Tail22
def root74Tail20 : List Nat := 4703 :: root74Tail21
def root74Tail19 : List Nat := 4637 :: root74Tail20
def root74Tail18 : List Nat := 4409 :: root74Tail19
def root74Tail17 : List Nat := 4049 :: root74Tail18
def root74Tail16 : List Nat := 3719 :: root74Tail17
def root74Tail15 : List Nat := 3299 :: root74Tail16
def root74Tail14 : List Nat := 3191 :: root74Tail15
def root74Tail13 : List Nat := 3137 :: root74Tail14
def root74Tail12 : List Nat := 2969 :: root74Tail13
def root74Tail11 : List Nat := 2459 :: root74Tail12
def root74Tail10 : List Nat := 2393 :: root74Tail11
def root74Tail9 : List Nat := 2333 :: root74Tail10
def root74Tail8 : List Nat := 2309 :: root74Tail9
def root74Tail7 : List Nat := 2087 :: root74Tail8
def root74Tail6 : List Nat := 1913 :: root74Tail7
def root74Tail5 : List Nat := 1493 :: root74Tail6
def root74Tail4 : List Nat := 1259 :: root74Tail5
def root74Tail3 : List Nat := 971 :: root74Tail4
def root74Tail2 : List Nat := 947 :: root74Tail3
def root74Tail1 : List Nat := 797 :: root74Tail2
def root74Tail0 : List Nat := 761 :: root74Tail1
def root74Candidates0 : List Nat := [5351, 13001]
set_option maxRecDepth 8192 in
theorem root74Candidates0_checked : candidates edge 4 389 761 root74Tail1 winner.bound = root74Candidates0 := by
  unfold candidates
  have row : edge 761 = (SortedNatTree.contains FastRows.row761) := by
    change Data.row761.contains = (SortedNatTree.contains FastRows.row761)
    exact FastRows.row761_checked.symm
  rw [row]
  decide
theorem root74Step0 : scan edge (search edge 3) 4 [389] 389 root74Tail0 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail1 winner := by
  rw [root74Tail0, scan, if_neg (by decide), root74Candidates0_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail1
    (if 3 ≤ root74Candidates0.length then search edge 3 [389, 761] root74Candidates0 1150 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates1 : List Nat := [2969, 3299]
set_option maxRecDepth 8192 in
theorem root74Candidates1_checked : candidates edge 4 389 797 root74Tail2 winner.bound = root74Candidates1 := by
  unfold candidates
  have row : edge 797 = (SortedNatTree.contains FastRows.row797) := by
    change Data.row797.contains = (SortedNatTree.contains FastRows.row797)
    exact FastRows.row797_checked.symm
  rw [row]
  decide
theorem root74Step1 : scan edge (search edge 3) 4 [389] 389 root74Tail1 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail2 winner := by
  rw [root74Tail1, scan, if_neg (by decide), root74Candidates1_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail2
    (if 3 ≤ root74Candidates1.length then search edge 3 [389, 797] root74Candidates1 1186 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates2 : List Nat := [4637, 6947, 12953, 18233, 23747]
set_option maxRecDepth 8192 in
theorem root74Candidates2_checked : candidates edge 4 389 947 root74Tail3 winner.bound = root74Candidates2 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root74Child2 : search edge 3 [389, 947] root74Candidates2 1336 winner = winner := by decide
theorem root74Step2 : scan edge (search edge 3) 4 [389] 389 root74Tail2 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail3 winner := by
  rw [root74Tail2, scan, if_neg (by decide), root74Candidates2_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail3
    (if 3 ≤ root74Candidates2.length then search edge 3 [389, 947] root74Candidates2 1336 winner else winner) = _
  rw [if_pos (by decide), root74Child2]
def root74Candidates3 : List Nat := [8123, 14009, 17093, 18587, 19571]
set_option maxRecDepth 8192 in
theorem root74Candidates3_checked : candidates edge 4 389 971 root74Tail4 winner.bound = root74Candidates3 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root74Child3 : search edge 3 [389, 971] root74Candidates3 1360 winner = winner := by decide
theorem root74Step3 : scan edge (search edge 3) 4 [389] 389 root74Tail3 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail4 winner := by
  rw [root74Tail3, scan, if_neg (by decide), root74Candidates3_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail4
    (if 3 ≤ root74Candidates3.length then search edge 3 [389, 971] root74Candidates3 1360 winner else winner) = _
  rw [if_pos (by decide), root74Child3]
def root74Candidates4 : List Nat := [20177]
set_option maxRecDepth 8192 in
theorem root74Candidates4_checked : candidates edge 4 389 1259 root74Tail5 winner.bound = root74Candidates4 := by
  unfold candidates
  have row : edge 1259 = (SortedNatTree.contains FastRows.row1259) := by
    change Data.row1259.contains = (SortedNatTree.contains FastRows.row1259)
    exact FastRows.row1259_checked.symm
  rw [row]
  decide
theorem root74Step4 : scan edge (search edge 3) 4 [389] 389 root74Tail4 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail5 winner := by
  rw [root74Tail4, scan, if_neg (by decide), root74Candidates4_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail5
    (if 3 ≤ root74Candidates4.length then search edge 3 [389, 1259] root74Candidates4 1648 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates5 : List Nat := [17891, 18401, 19571]
set_option maxRecDepth 8192 in
theorem root74Candidates5_checked : candidates edge 4 389 1493 root74Tail6 winner.bound = root74Candidates5 := by
  unfold candidates
  have row : edge 1493 = (SortedNatTree.contains FastRows.row1493) := by
    change Data.row1493.contains = (SortedNatTree.contains FastRows.row1493)
    exact FastRows.row1493_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root74Child5 : search edge 3 [389, 1493] root74Candidates5 1882 winner = winner := by decide
theorem root74Step5 : scan edge (search edge 3) 4 [389] 389 root74Tail5 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail6 winner := by
  rw [root74Tail5, scan, if_neg (by decide), root74Candidates5_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail6
    (if 3 ≤ root74Candidates5.length then search edge 3 [389, 1493] root74Candidates5 1882 winner else winner) = _
  rw [if_pos (by decide), root74Child5]
def root74Candidates6 : List Nat := [18839]
set_option maxRecDepth 8192 in
theorem root74Candidates6_checked : candidates edge 4 389 1913 root74Tail7 winner.bound = root74Candidates6 := by
  unfold candidates
  have row : edge 1913 = (SortedNatTree.contains FastRows.row1913) := by
    change Data.row1913.contains = (SortedNatTree.contains FastRows.row1913)
    exact FastRows.row1913_checked.symm
  rw [row]
  decide
theorem root74Step6 : scan edge (search edge 3) 4 [389] 389 root74Tail6 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail7 winner := by
  rw [root74Tail6, scan, if_neg (by decide), root74Candidates6_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail7
    (if 3 ≤ root74Candidates6.length then search edge 3 [389, 1913] root74Candidates6 2302 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates7 : List Nat := []
set_option maxRecDepth 8192 in
theorem root74Candidates7_checked : candidates edge 4 389 2087 root74Tail8 winner.bound = root74Candidates7 := by
  unfold candidates
  have row : edge 2087 = (SortedNatTree.contains FastRows.row2087) := by
    change Data.row2087.contains = (SortedNatTree.contains FastRows.row2087)
    exact FastRows.row2087_checked.symm
  rw [row]
  decide
theorem root74Step7 : scan edge (search edge 3) 4 [389] 389 root74Tail7 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail8 winner := by
  rw [root74Tail7, scan, if_neg (by decide), root74Candidates7_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail8
    (if 3 ≤ root74Candidates7.length then search edge 3 [389, 2087] root74Candidates7 2476 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates8 : List Nat := [6257, 6947]
set_option maxRecDepth 8192 in
theorem root74Candidates8_checked : candidates edge 4 389 2309 root74Tail9 winner.bound = root74Candidates8 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
theorem root74Step8 : scan edge (search edge 3) 4 [389] 389 root74Tail8 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail9 winner := by
  rw [root74Tail8, scan, if_neg (by decide), root74Candidates8_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail9
    (if 3 ≤ root74Candidates8.length then search edge 3 [389, 2309] root74Candidates8 2698 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates9 : List Nat := [3137, 14867, 19157]
set_option maxRecDepth 8192 in
theorem root74Candidates9_checked : candidates edge 4 389 2333 root74Tail10 winner.bound = root74Candidates9 := by
  unfold candidates
  have row : edge 2333 = (SortedNatTree.contains FastRows.row2333) := by
    change Data.row2333.contains = (SortedNatTree.contains FastRows.row2333)
    exact FastRows.row2333_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root74Child9 : search edge 3 [389, 2333] root74Candidates9 2722 winner = winner := by decide
theorem root74Step9 : scan edge (search edge 3) 4 [389] 389 root74Tail9 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail10 winner := by
  rw [root74Tail9, scan, if_neg (by decide), root74Candidates9_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail10
    (if 3 ≤ root74Candidates9.length then search edge 3 [389, 2333] root74Candidates9 2722 winner else winner) = _
  rw [if_pos (by decide), root74Child9]
def root74Candidates10 : List Nat := [17387]
set_option maxRecDepth 8192 in
theorem root74Candidates10_checked : candidates edge 4 389 2393 root74Tail11 winner.bound = root74Candidates10 := by
  unfold candidates
  have row : edge 2393 = (SortedNatTree.contains FastRows.row2393) := by
    change Data.row2393.contains = (SortedNatTree.contains FastRows.row2393)
    exact FastRows.row2393_checked.symm
  rw [row]
  decide
theorem root74Step10 : scan edge (search edge 3) 4 [389] 389 root74Tail10 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail11 winner := by
  rw [root74Tail10, scan, if_neg (by decide), root74Candidates10_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail11
    (if 3 ≤ root74Candidates10.length then search edge 3 [389, 2393] root74Candidates10 2782 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates11 : List Nat := [14153]
set_option maxRecDepth 8192 in
theorem root74Candidates11_checked : candidates edge 4 389 2459 root74Tail12 winner.bound = root74Candidates11 := by
  unfold candidates
  have row : edge 2459 = (SortedNatTree.contains FastRows.row2459) := by
    change Data.row2459.contains = (SortedNatTree.contains FastRows.row2459)
    exact FastRows.row2459_checked.symm
  rw [row]
  decide
theorem root74Step11 : scan edge (search edge 3) 4 [389] 389 root74Tail11 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail12 winner := by
  rw [root74Tail11, scan, if_neg (by decide), root74Candidates11_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail12
    (if 3 ≤ root74Candidates11.length then search edge 3 [389, 2459] root74Candidates11 2848 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates12 : List Nat := [4703, 12953]
set_option maxRecDepth 8192 in
theorem root74Candidates12_checked : candidates edge 4 389 2969 root74Tail13 winner.bound = root74Candidates12 := by
  unfold candidates
  have row : edge 2969 = (SortedNatTree.contains FastRows.row2969) := by
    change Data.row2969.contains = (SortedNatTree.contains FastRows.row2969)
    exact FastRows.row2969_checked.symm
  rw [row]
  decide
theorem root74Step12 : scan edge (search edge 3) 4 [389] 389 root74Tail12 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail13 winner := by
  rw [root74Tail12, scan, if_neg (by decide), root74Candidates12_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail13
    (if 3 ≤ root74Candidates12.length then search edge 3 [389, 2969] root74Candidates12 3358 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates13 : List Nat := [8513, 14867, 17387]
set_option maxRecDepth 8192 in
theorem root74Candidates13_checked : candidates edge 4 389 3137 root74Tail14 winner.bound = root74Candidates13 := by
  unfold candidates
  have row : edge 3137 = (SortedNatTree.contains FastRows.row3137) := by
    change Data.row3137.contains = (SortedNatTree.contains FastRows.row3137)
    exact FastRows.row3137_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root74Child13 : search edge 3 [389, 3137] root74Candidates13 3526 winner = winner := by decide
theorem root74Step13 : scan edge (search edge 3) 4 [389] 389 root74Tail13 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail14 winner := by
  rw [root74Tail13, scan, if_neg (by decide), root74Candidates13_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail14
    (if 3 ≤ root74Candidates13.length then search edge 3 [389, 3137] root74Candidates13 3526 winner else winner) = _
  rw [if_pos (by decide), root74Child13]
def root74Candidates14 : List Nat := [7703, 7937, 17891]
set_option maxRecDepth 8192 in
theorem root74Candidates14_checked : candidates edge 4 389 3191 root74Tail15 winner.bound = root74Candidates14 := by
  unfold candidates
  have row : edge 3191 = (SortedNatTree.contains FastRows.row3191) := by
    change Data.row3191.contains = (SortedNatTree.contains FastRows.row3191)
    exact FastRows.row3191_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root74Child14 : search edge 3 [389, 3191] root74Candidates14 3580 winner = winner := by decide
theorem root74Step14 : scan edge (search edge 3) 4 [389] 389 root74Tail14 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail15 winner := by
  rw [root74Tail14, scan, if_neg (by decide), root74Candidates14_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail15
    (if 3 ≤ root74Candidates14.length then search edge 3 [389, 3191] root74Candidates14 3580 winner else winner) = _
  rw [if_pos (by decide), root74Child14]
def root74Candidates15 : List Nat := [7499]
set_option maxRecDepth 8192 in
theorem root74Candidates15_checked : candidates edge 4 389 3299 root74Tail16 winner.bound = root74Candidates15 := by
  unfold candidates
  have row : edge 3299 = (SortedNatTree.contains FastRows.row3299) := by
    change Data.row3299.contains = (SortedNatTree.contains FastRows.row3299)
    exact FastRows.row3299_checked.symm
  rw [row]
  decide
theorem root74Step15 : scan edge (search edge 3) 4 [389] 389 root74Tail15 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail16 winner := by
  rw [root74Tail15, scan, if_neg (by decide), root74Candidates15_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail16
    (if 3 ≤ root74Candidates15.length then search edge 3 [389, 3299] root74Candidates15 3688 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates16 : List Nat := [19157]
set_option maxRecDepth 8192 in
theorem root74Candidates16_checked : candidates edge 4 389 3719 root74Tail17 winner.bound = root74Candidates16 := by
  unfold candidates
  have row : edge 3719 = (SortedNatTree.contains FastRows.row3719) := by
    change Data.row3719.contains = (SortedNatTree.contains FastRows.row3719)
    exact FastRows.row3719_checked.symm
  rw [row]
  decide
theorem root74Step16 : scan edge (search edge 3) 4 [389] 389 root74Tail16 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail17 winner := by
  rw [root74Tail16, scan, if_neg (by decide), root74Candidates16_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail17
    (if 3 ≤ root74Candidates16.length then search edge 3 [389, 3719] root74Candidates16 4108 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates17 : List Nat := [7703]
set_option maxRecDepth 8192 in
theorem root74Candidates17_checked : candidates edge 4 389 4049 root74Tail18 winner.bound = root74Candidates17 := by
  unfold candidates
  have row : edge 4049 = (SortedNatTree.contains FastRows.row4049) := by
    change Data.row4049.contains = (SortedNatTree.contains FastRows.row4049)
    exact FastRows.row4049_checked.symm
  rw [row]
  decide
theorem root74Step17 : scan edge (search edge 3) 4 [389] 389 root74Tail17 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail18 winner := by
  rw [root74Tail17, scan, if_neg (by decide), root74Candidates17_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail18
    (if 3 ≤ root74Candidates17.length then search edge 3 [389, 4049] root74Candidates17 4438 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates18 : List Nat := [14153]
set_option maxRecDepth 8192 in
theorem root74Candidates18_checked : candidates edge 4 389 4409 root74Tail19 winner.bound = root74Candidates18 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root74Step18 : scan edge (search edge 3) 4 [389] 389 root74Tail18 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail19 winner := by
  rw [root74Tail18, scan, if_neg (by decide), root74Candidates18_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail19
    (if 3 ≤ root74Candidates18.length then search edge 3 [389, 4409] root74Candidates18 4798 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates19 : List Nat := []
set_option maxRecDepth 8192 in
theorem root74Candidates19_checked : candidates edge 4 389 4637 root74Tail20 winner.bound = root74Candidates19 := by
  unfold candidates
  have row : edge 4637 = (SortedNatTree.contains FastRows.row4637) := by
    change Data.row4637.contains = (SortedNatTree.contains FastRows.row4637)
    exact FastRows.row4637_checked.symm
  rw [row]
  decide
theorem root74Step19 : scan edge (search edge 3) 4 [389] 389 root74Tail19 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail20 winner := by
  rw [root74Tail19, scan, if_neg (by decide), root74Candidates19_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail20
    (if 3 ≤ root74Candidates19.length then search edge 3 [389, 4637] root74Candidates19 5026 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates20 : List Nat := [6257]
set_option maxRecDepth 8192 in
theorem root74Candidates20_checked : candidates edge 4 389 4703 root74Tail21 winner.bound = root74Candidates20 := by
  unfold candidates
  have row : edge 4703 = (SortedNatTree.contains FastRows.row4703) := by
    change Data.row4703.contains = (SortedNatTree.contains FastRows.row4703)
    exact FastRows.row4703_checked.symm
  rw [row]
  decide
theorem root74Step20 : scan edge (search edge 3) 4 [389] 389 root74Tail20 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail21 winner := by
  rw [root74Tail20, scan, if_neg (by decide), root74Candidates20_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail21
    (if 3 ≤ root74Candidates20.length then search edge 3 [389, 4703] root74Candidates20 5092 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates21 : List Nat := []
set_option maxRecDepth 8192 in
theorem root74Candidates21_checked : candidates edge 4 389 4733 root74Tail22 winner.bound = root74Candidates21 := by
  unfold candidates
  have row : edge 4733 = (SortedNatTree.contains FastRows.row4733) := by
    change Data.row4733.contains = (SortedNatTree.contains FastRows.row4733)
    exact FastRows.row4733_checked.symm
  rw [row]
  decide
theorem root74Step21 : scan edge (search edge 3) 4 [389] 389 root74Tail21 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail22 winner := by
  rw [root74Tail21, scan, if_neg (by decide), root74Candidates21_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail22
    (if 3 ≤ root74Candidates21.length then search edge 3 [389, 4733] root74Candidates21 5122 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates22 : List Nat := [13229, 17093]
set_option maxRecDepth 8192 in
theorem root74Candidates22_checked : candidates edge 4 389 5231 root74Tail23 winner.bound = root74Candidates22 := by
  unfold candidates
  have row : edge 5231 = (SortedNatTree.contains FastRows.row5231) := by
    change Data.row5231.contains = (SortedNatTree.contains FastRows.row5231)
    exact FastRows.row5231_checked.symm
  rw [row]
  decide
theorem root74Step22 : scan edge (search edge 3) 4 [389] 389 root74Tail22 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail23 winner := by
  rw [root74Tail22, scan, if_neg (by decide), root74Candidates22_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail23
    (if 3 ≤ root74Candidates22.length then search edge 3 [389, 5231] root74Candidates22 5620 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates23 : List Nat := [11321, 20177]
set_option maxRecDepth 8192 in
theorem root74Candidates23_checked : candidates edge 4 389 5351 root74Tail24 winner.bound = root74Candidates23 := by
  unfold candidates
  have row : edge 5351 = (SortedNatTree.contains FastRows.row5351) := by
    change Data.row5351.contains = (SortedNatTree.contains FastRows.row5351)
    exact FastRows.row5351_checked.symm
  rw [row]
  decide
theorem root74Step23 : scan edge (search edge 3) 4 [389] 389 root74Tail23 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail24 winner := by
  rw [root74Tail23, scan, if_neg (by decide), root74Candidates23_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail24
    (if 3 ≤ root74Candidates23.length then search edge 3 [389, 5351] root74Candidates23 5740 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates24 : List Nat := [14867, 18401]
set_option maxRecDepth 8192 in
theorem root74Candidates24_checked : candidates edge 4 389 6029 root74Tail25 winner.bound = root74Candidates24 := by
  unfold candidates
  have row : edge 6029 = (SortedNatTree.contains FastRows.row6029) := by
    change Data.row6029.contains = (SortedNatTree.contains FastRows.row6029)
    exact FastRows.row6029_checked.symm
  rw [row]
  decide
theorem root74Step24 : scan edge (search edge 3) 4 [389] 389 root74Tail24 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail25 winner := by
  rw [root74Tail24, scan, if_neg (by decide), root74Candidates24_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail25
    (if 3 ≤ root74Candidates24.length then search edge 3 [389, 6029] root74Candidates24 6418 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates25 : List Nat := []
set_option maxRecDepth 8192 in
theorem root74Candidates25_checked : candidates edge 4 389 6089 root74Tail26 winner.bound = root74Candidates25 := by
  unfold candidates
  have row : edge 6089 = (SortedNatTree.contains FastRows.row6089) := by
    change Data.row6089.contains = (SortedNatTree.contains FastRows.row6089)
    exact FastRows.row6089_checked.symm
  rw [row]
  decide
theorem root74Step25 : scan edge (search edge 3) 4 [389] 389 root74Tail25 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail26 winner := by
  rw [root74Tail25, scan, if_neg (by decide), root74Candidates25_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail26
    (if 3 ≤ root74Candidates25.length then search edge 3 [389, 6089] root74Candidates25 6478 winner else winner) = _
  rw [if_neg (by decide)]
def root74Candidates26 : List Nat := []
set_option maxRecDepth 8192 in
theorem root74Candidates26_checked : candidates edge 4 389 6257 root74Tail27 winner.bound = root74Candidates26 := by
  unfold candidates
  have row : edge 6257 = (SortedNatTree.contains FastRows.row6257) := by
    change Data.row6257.contains = (SortedNatTree.contains FastRows.row6257)
    exact FastRows.row6257_checked.symm
  rw [row]
  decide
theorem root74Step26 : scan edge (search edge 3) 4 [389] 389 root74Tail26 winner =
    scan edge (search edge 3) 4 [389] 389 root74Tail27 winner := by
  rw [root74Tail26, scan, if_neg (by decide), root74Candidates26_checked]
  change scan edge (search edge 3) 4 [389] 389 root74Tail27
    (if 3 ≤ root74Candidates26.length then search edge 3 [389, 6257] root74Candidates26 6646 winner else winner) = _
  rw [if_neg (by decide)]
theorem root74Scan27 : scan edge (search edge 3) 4 [389] 389 root74Tail27 winner = winner := by rw [root74Tail27, scan, if_pos (by decide)]
theorem root74Scan26 : scan edge (search edge 3) 4 [389] 389 root74Tail26 winner = winner :=
  (root74Step26).trans root74Scan27
theorem root74Scan25 : scan edge (search edge 3) 4 [389] 389 root74Tail25 winner = winner :=
  (root74Step25).trans root74Scan26
theorem root74Scan24 : scan edge (search edge 3) 4 [389] 389 root74Tail24 winner = winner :=
  (root74Step24).trans root74Scan25
theorem root74Scan23 : scan edge (search edge 3) 4 [389] 389 root74Tail23 winner = winner :=
  (root74Step23).trans root74Scan24
theorem root74Scan22 : scan edge (search edge 3) 4 [389] 389 root74Tail22 winner = winner :=
  (root74Step22).trans root74Scan23
theorem root74Scan21 : scan edge (search edge 3) 4 [389] 389 root74Tail21 winner = winner :=
  (root74Step21).trans root74Scan22
theorem root74Scan20 : scan edge (search edge 3) 4 [389] 389 root74Tail20 winner = winner :=
  (root74Step20).trans root74Scan21
theorem root74Scan19 : scan edge (search edge 3) 4 [389] 389 root74Tail19 winner = winner :=
  (root74Step19).trans root74Scan20
theorem root74Scan18 : scan edge (search edge 3) 4 [389] 389 root74Tail18 winner = winner :=
  (root74Step18).trans root74Scan19
theorem root74Scan17 : scan edge (search edge 3) 4 [389] 389 root74Tail17 winner = winner :=
  (root74Step17).trans root74Scan18
theorem root74Scan16 : scan edge (search edge 3) 4 [389] 389 root74Tail16 winner = winner :=
  (root74Step16).trans root74Scan17
theorem root74Scan15 : scan edge (search edge 3) 4 [389] 389 root74Tail15 winner = winner :=
  (root74Step15).trans root74Scan16
theorem root74Scan14 : scan edge (search edge 3) 4 [389] 389 root74Tail14 winner = winner :=
  (root74Step14).trans root74Scan15
theorem root74Scan13 : scan edge (search edge 3) 4 [389] 389 root74Tail13 winner = winner :=
  (root74Step13).trans root74Scan14
theorem root74Scan12 : scan edge (search edge 3) 4 [389] 389 root74Tail12 winner = winner :=
  (root74Step12).trans root74Scan13
theorem root74Scan11 : scan edge (search edge 3) 4 [389] 389 root74Tail11 winner = winner :=
  (root74Step11).trans root74Scan12
theorem root74Scan10 : scan edge (search edge 3) 4 [389] 389 root74Tail10 winner = winner :=
  (root74Step10).trans root74Scan11
theorem root74Scan9 : scan edge (search edge 3) 4 [389] 389 root74Tail9 winner = winner :=
  (root74Step9).trans root74Scan10
theorem root74Scan8 : scan edge (search edge 3) 4 [389] 389 root74Tail8 winner = winner :=
  (root74Step8).trans root74Scan9
theorem root74Scan7 : scan edge (search edge 3) 4 [389] 389 root74Tail7 winner = winner :=
  (root74Step7).trans root74Scan8
theorem root74Scan6 : scan edge (search edge 3) 4 [389] 389 root74Tail6 winner = winner :=
  (root74Step6).trans root74Scan7
theorem root74Scan5 : scan edge (search edge 3) 4 [389] 389 root74Tail5 winner = winner :=
  (root74Step5).trans root74Scan6
theorem root74Scan4 : scan edge (search edge 3) 4 [389] 389 root74Tail4 winner = winner :=
  (root74Step4).trans root74Scan5
theorem root74Scan3 : scan edge (search edge 3) 4 [389] 389 root74Tail3 winner = winner :=
  (root74Step3).trans root74Scan4
theorem root74Scan2 : scan edge (search edge 3) 4 [389] 389 root74Tail2 winner = winner :=
  (root74Step2).trans root74Scan3
theorem root74Scan1 : scan edge (search edge 3) 4 [389] 389 root74Tail1 winner = winner :=
  (root74Step1).trans root74Scan2
theorem root74Scan0 : scan edge (search edge 3) 4 [389] 389 root74Tail0 winner = winner :=
  (root74Step0).trans root74Scan1
set_option maxRecDepth 4096 in
theorem root74Branch : search edge 4 [389] root74Candidates 389 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [389] 389 root74Tail0 winner = winner
  exact root74Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
