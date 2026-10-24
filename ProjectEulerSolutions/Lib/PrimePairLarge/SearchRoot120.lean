import ProjectEulerSolutions.Lib.PrimePairLarge.SearchSetup
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root120Candidates : List Nat := Data.row677
set_option maxRecDepth 8192 in
theorem root120Candidates_checked : candidates edge 5 0 677 tail121 winner.bound = root120Candidates := by
  apply candidates_eq_row edge 5 0 677 tail121 root120Candidates winner.bound tail121_ordered
  · decide
  · decide
  · funext q
    rfl
def root120Tail75 : List Nat := []
def root120Tail74 : List Nat := 24317 :: root120Tail75
def root120Tail73 : List Nat := 22961 :: root120Tail74
def root120Tail72 : List Nat := 22943 :: root120Tail73
def root120Tail71 : List Nat := 21569 :: root120Tail72
def root120Tail70 : List Nat := 20483 :: root120Tail71
def root120Tail69 : List Nat := 20369 :: root120Tail70
def root120Tail68 : List Nat := 19973 :: root120Tail69
def root120Tail67 : List Nat := 19739 :: root120Tail68
def root120Tail66 : List Nat := 19697 :: root120Tail67
def root120Tail65 : List Nat := 19571 :: root120Tail66
def root120Tail64 : List Nat := 19391 :: root120Tail65
def root120Tail63 : List Nat := 19373 :: root120Tail64
def root120Tail62 : List Nat := 17921 :: root120Tail63
def root120Tail61 : List Nat := 17627 :: root120Tail62
def root120Tail60 : List Nat := 17189 :: root120Tail61
def root120Tail59 : List Nat := 16937 :: root120Tail60
def root120Tail58 : List Nat := 16487 :: root120Tail59
def root120Tail57 : List Nat := 15971 :: root120Tail58
def root120Tail56 : List Nat := 15773 :: root120Tail57
def root120Tail55 : List Nat := 15227 :: root120Tail56
def root120Tail54 : List Nat := 15173 :: root120Tail55
def root120Tail53 : List Nat := 15077 :: root120Tail54
def root120Tail52 : List Nat := 15017 :: root120Tail53
def root120Tail51 : List Nat := 14321 :: root120Tail52
def root120Tail50 : List Nat := 14057 :: root120Tail51
def root120Tail49 : List Nat := 13829 :: root120Tail50
def root120Tail48 : List Nat := 13577 :: root120Tail49
def root120Tail47 : List Nat := 13421 :: root120Tail48
def root120Tail46 : List Nat := 13049 :: root120Tail47
def root120Tail45 : List Nat := 13043 :: root120Tail46
def root120Tail44 : List Nat := 12527 :: root120Tail45
def root120Tail43 : List Nat := 12473 :: root120Tail44
def root120Tail42 : List Nat := 12329 :: root120Tail43
def root120Tail41 : List Nat := 12263 :: root120Tail42
def root120Tail40 : List Nat := 12251 :: root120Tail41
def root120Tail39 : List Nat := 12119 :: root120Tail40
def root120Tail38 : List Nat := 10667 :: root120Tail39
def root120Tail37 : List Nat := 9743 :: root120Tail38
def root120Tail36 : List Nat := 9629 :: root120Tail37
def root120Tail35 : List Nat := 9467 :: root120Tail36
def root120Tail34 : List Nat := 8963 :: root120Tail35
def root120Tail33 : List Nat := 8147 :: root120Tail34
def root120Tail32 : List Nat := 8039 :: root120Tail33
def root120Tail31 : List Nat := 7949 :: root120Tail32
def root120Tail30 : List Nat := 7517 :: root120Tail31
def root120Tail29 : List Nat := 7451 :: root120Tail30
def root120Tail28 : List Nat := 7349 :: root120Tail29
def root120Tail27 : List Nat := 6971 :: root120Tail28
def root120Tail26 : List Nat := 6491 :: root120Tail27
def root120Tail25 : List Nat := 6257 :: root120Tail26
def root120Tail24 : List Nat := 6173 :: root120Tail25
def root120Tail23 : List Nat := 6101 :: root120Tail24
def root120Tail22 : List Nat := 5987 :: root120Tail23
def root120Tail21 : List Nat := 5861 :: root120Tail22
def root120Tail20 : List Nat := 5807 :: root120Tail21
def root120Tail19 : List Nat := 5519 :: root120Tail20
def root120Tail18 : List Nat := 5399 :: root120Tail19
def root120Tail17 : List Nat := 5393 :: root120Tail18
def root120Tail16 : List Nat := 5387 :: root120Tail17
def root120Tail15 : List Nat := 5273 :: root120Tail16
def root120Tail14 : List Nat := 4451 :: root120Tail15
def root120Tail13 : List Nat := 4409 :: root120Tail14
def root120Tail12 : List Nat := 4073 :: root120Tail13
def root120Tail11 : List Nat := 3863 :: root120Tail12
def root120Tail10 : List Nat := 3203 :: root120Tail11
def root120Tail9 : List Nat := 2837 :: root120Tail10
def root120Tail8 : List Nat := 2687 :: root120Tail9
def root120Tail7 : List Nat := 2309 :: root120Tail8
def root120Tail6 : List Nat := 1871 :: root120Tail7
def root120Tail5 : List Nat := 1493 :: root120Tail6
def root120Tail4 : List Nat := 1451 :: root120Tail5
def root120Tail3 : List Nat := 1409 :: root120Tail4
def root120Tail2 : List Nat := 1283 :: root120Tail3
def root120Tail1 : List Nat := 1217 :: root120Tail2
def root120Tail0 : List Nat := 827 :: root120Tail1
def root120Candidates0 : List Nat := [1871]
set_option maxRecDepth 8192 in
theorem root120Candidates0_checked : candidates edge 4 677 827 root120Tail1 winner.bound = root120Candidates0 := by
  unfold candidates
  have row : edge 827 = (SortedNatTree.contains FastRows.row827) := by
    change Data.row827.contains = (SortedNatTree.contains FastRows.row827)
    exact FastRows.row827_checked.symm
  rw [row]
  decide
theorem root120Step0 : scan edge (search edge 3) 4 [677] 677 root120Tail0 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail1 winner := by
  rw [root120Tail0, scan, if_neg (by decide), root120Candidates0_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail1
    (if 3 ≤ root120Candidates0.length then search edge 3 [677, 827] root120Candidates0 1504 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates1 : List Nat := [2309, 5393, 6173, 16937]
set_option maxRecDepth 8192 in
theorem root120Candidates1_checked : candidates edge 4 677 1217 root120Tail2 winner.bound = root120Candidates1 := by
  unfold candidates
  have row : edge 1217 = (SortedNatTree.contains FastRows.row1217) := by
    change Data.row1217.contains = (SortedNatTree.contains FastRows.row1217)
    exact FastRows.row1217_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root120Child1 : search edge 3 [677, 1217] root120Candidates1 1894 winner = winner := by decide
theorem root120Step1 : scan edge (search edge 3) 4 [677] 677 root120Tail1 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail2 winner := by
  rw [root120Tail1, scan, if_neg (by decide), root120Candidates1_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail2
    (if 3 ≤ root120Candidates1.length then search edge 3 [677, 1217] root120Candidates1 1894 winner else winner) = _
  rw [if_pos (by decide), root120Child1]
def root120Candidates2 : List Nat := [5807, 7949, 15017, 19739]
set_option maxRecDepth 8192 in
theorem root120Candidates2_checked : candidates edge 4 677 1283 root120Tail3 winner.bound = root120Candidates2 := by
  unfold candidates
  have row : edge 1283 = (SortedNatTree.contains FastRows.row1283) := by
    change Data.row1283.contains = (SortedNatTree.contains FastRows.row1283)
    exact FastRows.row1283_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root120Child2 : search edge 3 [677, 1283] root120Candidates2 1960 winner = winner := by decide
theorem root120Step2 : scan edge (search edge 3) 4 [677] 677 root120Tail2 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail3 winner := by
  rw [root120Tail2, scan, if_neg (by decide), root120Candidates2_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail3
    (if 3 ≤ root120Candidates2.length then search edge 3 [677, 1283] root120Candidates2 1960 winner else winner) = _
  rw [if_pos (by decide), root120Child2]
def root120Candidates3 : List Nat := [5387]
set_option maxRecDepth 8192 in
theorem root120Candidates3_checked : candidates edge 4 677 1409 root120Tail4 winner.bound = root120Candidates3 := by
  unfold candidates
  have row : edge 1409 = (SortedNatTree.contains FastRows.row1409) := by
    change Data.row1409.contains = (SortedNatTree.contains FastRows.row1409)
    exact FastRows.row1409_checked.symm
  rw [row]
  decide
theorem root120Step3 : scan edge (search edge 3) 4 [677] 677 root120Tail3 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail4 winner := by
  rw [root120Tail3, scan, if_neg (by decide), root120Candidates3_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail4
    (if 3 ≤ root120Candidates3.length then search edge 3 [677, 1409] root120Candidates3 2086 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates4 : List Nat := [15227]
set_option maxRecDepth 8192 in
theorem root120Candidates4_checked : candidates edge 4 677 1451 root120Tail5 winner.bound = root120Candidates4 := by
  unfold candidates
  have row : edge 1451 = (SortedNatTree.contains FastRows.row1451) := by
    change Data.row1451.contains = (SortedNatTree.contains FastRows.row1451)
    exact FastRows.row1451_checked.symm
  rw [row]
  decide
theorem root120Step4 : scan edge (search edge 3) 4 [677] 677 root120Tail4 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail5 winner := by
  rw [root120Tail4, scan, if_neg (by decide), root120Candidates4_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail5
    (if 3 ≤ root120Candidates4.length then search edge 3 [677, 1451] root120Candidates4 2128 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates5 : List Nat := [17921, 19571]
set_option maxRecDepth 8192 in
theorem root120Candidates5_checked : candidates edge 4 677 1493 root120Tail6 winner.bound = root120Candidates5 := by
  unfold candidates
  have row : edge 1493 = (SortedNatTree.contains FastRows.row1493) := by
    change Data.row1493.contains = (SortedNatTree.contains FastRows.row1493)
    exact FastRows.row1493_checked.symm
  rw [row]
  decide
theorem root120Step5 : scan edge (search edge 3) 4 [677] 677 root120Tail5 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail6 winner := by
  rw [root120Tail5, scan, if_neg (by decide), root120Candidates5_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail6
    (if 3 ≤ root120Candidates5.length then search edge 3 [677, 1493] root120Candidates5 2170 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates6 : List Nat := [12119, 17921]
set_option maxRecDepth 8192 in
theorem root120Candidates6_checked : candidates edge 4 677 1871 root120Tail7 winner.bound = root120Candidates6 := by
  unfold candidates
  have row : edge 1871 = (SortedNatTree.contains FastRows.row1871) := by
    change Data.row1871.contains = (SortedNatTree.contains FastRows.row1871)
    exact FastRows.row1871_checked.symm
  rw [row]
  decide
theorem root120Step6 : scan edge (search edge 3) 4 [677] 677 root120Tail6 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail7 winner := by
  rw [root120Tail6, scan, if_neg (by decide), root120Candidates6_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail7
    (if 3 ≤ root120Candidates6.length then search edge 3 [677, 1871] root120Candidates6 2548 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates7 : List Nat := [2687, 4451, 6257, 19697, 20369]
set_option maxRecDepth 8192 in
theorem root120Candidates7_checked : candidates edge 4 677 2309 root120Tail8 winner.bound = root120Candidates7 := by
  unfold candidates
  have row : edge 2309 = (SortedNatTree.contains FastRows.row2309) := by
    change Data.row2309.contains = (SortedNatTree.contains FastRows.row2309)
    exact FastRows.row2309_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root120Child7 : search edge 3 [677, 2309] root120Candidates7 2986 winner = winner := by decide
theorem root120Step7 : scan edge (search edge 3) 4 [677] 677 root120Tail7 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail8 winner := by
  rw [root120Tail7, scan, if_neg (by decide), root120Candidates7_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail8
    (if 3 ≤ root120Candidates7.length then search edge 3 [677, 2309] root120Candidates7 2986 winner else winner) = _
  rw [if_pos (by decide), root120Child7]
def root120Candidates8 : List Nat := [8039, 15971]
set_option maxRecDepth 8192 in
theorem root120Candidates8_checked : candidates edge 4 677 2687 root120Tail9 winner.bound = root120Candidates8 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
theorem root120Step8 : scan edge (search edge 3) 4 [677] 677 root120Tail8 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail9 winner := by
  rw [root120Tail8, scan, if_neg (by decide), root120Candidates8_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail9
    (if 3 ≤ root120Candidates8.length then search edge 3 [677, 2687] root120Candidates8 3364 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates9 : List Nat := [12329]
set_option maxRecDepth 8192 in
theorem root120Candidates9_checked : candidates edge 4 677 2837 root120Tail10 winner.bound = root120Candidates9 := by
  unfold candidates
  have row : edge 2837 = (SortedNatTree.contains FastRows.row2837) := by
    change Data.row2837.contains = (SortedNatTree.contains FastRows.row2837)
    exact FastRows.row2837_checked.symm
  rw [row]
  decide
theorem root120Step9 : scan edge (search edge 3) 4 [677] 677 root120Tail9 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail10 winner := by
  rw [root120Tail9, scan, if_neg (by decide), root120Candidates9_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail10
    (if 3 ≤ root120Candidates9.length then search edge 3 [677, 2837] root120Candidates9 3514 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates10 : List Nat := [6101, 7949, 13043]
set_option maxRecDepth 8192 in
theorem root120Candidates10_checked : candidates edge 4 677 3203 root120Tail11 winner.bound = root120Candidates10 := by
  unfold candidates
  have row : edge 3203 = (SortedNatTree.contains FastRows.row3203) := by
    change Data.row3203.contains = (SortedNatTree.contains FastRows.row3203)
    exact FastRows.row3203_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root120Child10 : search edge 3 [677, 3203] root120Candidates10 3880 winner = winner := by decide
theorem root120Step10 : scan edge (search edge 3) 4 [677] 677 root120Tail10 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail11 winner := by
  rw [root120Tail10, scan, if_neg (by decide), root120Candidates10_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail11
    (if 3 ≤ root120Candidates10.length then search edge 3 [677, 3203] root120Candidates10 3880 winner else winner) = _
  rw [if_pos (by decide), root120Child10]
def root120Candidates11 : List Nat := [4451]
set_option maxRecDepth 8192 in
theorem root120Candidates11_checked : candidates edge 4 677 3863 root120Tail12 winner.bound = root120Candidates11 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
theorem root120Step11 : scan edge (search edge 3) 4 [677] 677 root120Tail11 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail12 winner := by
  rw [root120Tail11, scan, if_neg (by decide), root120Candidates11_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail12
    (if 3 ≤ root120Candidates11.length then search edge 3 [677, 3863] root120Candidates11 4540 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates12 : List Nat := [5393, 17627, 19697]
set_option maxRecDepth 8192 in
theorem root120Candidates12_checked : candidates edge 4 677 4073 root120Tail13 winner.bound = root120Candidates12 := by
  unfold candidates
  have row : edge 4073 = (SortedNatTree.contains FastRows.row4073) := by
    change Data.row4073.contains = (SortedNatTree.contains FastRows.row4073)
    exact FastRows.row4073_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root120Child12 : search edge 3 [677, 4073] root120Candidates12 4750 winner = winner := by decide
theorem root120Step12 : scan edge (search edge 3) 4 [677] 677 root120Tail12 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail13 winner := by
  rw [root120Tail12, scan, if_neg (by decide), root120Candidates12_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail13
    (if 3 ≤ root120Candidates12.length then search edge 3 [677, 4073] root120Candidates12 4750 winner else winner) = _
  rw [if_pos (by decide), root120Child12]
def root120Candidates13 : List Nat := []
set_option maxRecDepth 8192 in
theorem root120Candidates13_checked : candidates edge 4 677 4409 root120Tail14 winner.bound = root120Candidates13 := by
  unfold candidates
  have row : edge 4409 = (SortedNatTree.contains FastRows.row4409) := by
    change Data.row4409.contains = (SortedNatTree.contains FastRows.row4409)
    exact FastRows.row4409_checked.symm
  rw [row]
  decide
theorem root120Step13 : scan edge (search edge 3) 4 [677] 677 root120Tail13 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail14 winner := by
  rw [root120Tail13, scan, if_neg (by decide), root120Candidates13_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail14
    (if 3 ≤ root120Candidates13.length then search edge 3 [677, 4409] root120Candidates13 5086 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates14 : List Nat := [6257, 8963, 14057, 20369]
set_option maxRecDepth 8192 in
theorem root120Candidates14_checked : candidates edge 4 677 4451 root120Tail15 winner.bound = root120Candidates14 := by
  unfold candidates
  have row : edge 4451 = (SortedNatTree.contains FastRows.row4451) := by
    change Data.row4451.contains = (SortedNatTree.contains FastRows.row4451)
    exact FastRows.row4451_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root120Child14 : search edge 3 [677, 4451] root120Candidates14 5128 winner = winner := by decide
theorem root120Step14 : scan edge (search edge 3) 4 [677] 677 root120Tail14 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail15 winner := by
  rw [root120Tail14, scan, if_neg (by decide), root120Candidates14_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail15
    (if 3 ≤ root120Candidates14.length then search edge 3 [677, 4451] root120Candidates14 5128 winner else winner) = _
  rw [if_pos (by decide), root120Child14]
def root120Candidates15 : List Nat := [13043, 14057]
set_option maxRecDepth 8192 in
theorem root120Candidates15_checked : candidates edge 4 677 5273 root120Tail16 winner.bound = root120Candidates15 := by
  unfold candidates
  have row : edge 5273 = (SortedNatTree.contains FastRows.row5273) := by
    change Data.row5273.contains = (SortedNatTree.contains FastRows.row5273)
    exact FastRows.row5273_checked.symm
  rw [row]
  decide
theorem root120Step15 : scan edge (search edge 3) 4 [677] 677 root120Tail15 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail16 winner := by
  rw [root120Tail15, scan, if_neg (by decide), root120Candidates15_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail16
    (if 3 ≤ root120Candidates15.length then search edge 3 [677, 5273] root120Candidates15 5950 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates16 : List Nat := [5399, 5807, 9467, 17189]
set_option maxRecDepth 8192 in
theorem root120Candidates16_checked : candidates edge 4 677 5387 root120Tail17 winner.bound = root120Candidates16 := by
  unfold candidates
  have row : edge 5387 = (SortedNatTree.contains FastRows.row5387) := by
    change Data.row5387.contains = (SortedNatTree.contains FastRows.row5387)
    exact FastRows.row5387_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root120Child16 : search edge 3 [677, 5387] root120Candidates16 6064 winner = winner := by decide
theorem root120Step16 : scan edge (search edge 3) 4 [677] 677 root120Tail16 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail17 winner := by
  rw [root120Tail16, scan, if_neg (by decide), root120Candidates16_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail17
    (if 3 ≤ root120Candidates16.length then search edge 3 [677, 5387] root120Candidates16 6064 winner else winner) = _
  rw [if_pos (by decide), root120Child16]
def root120Candidates17 : List Nat := []
set_option maxRecDepth 8192 in
theorem root120Candidates17_checked : candidates edge 4 677 5393 root120Tail18 winner.bound = root120Candidates17 := by
  unfold candidates
  have row : edge 5393 = (SortedNatTree.contains FastRows.row5393) := by
    change Data.row5393.contains = (SortedNatTree.contains FastRows.row5393)
    exact FastRows.row5393_checked.symm
  rw [row]
  decide
theorem root120Step17 : scan edge (search edge 3) 4 [677] 677 root120Tail17 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail18 winner := by
  rw [root120Tail17, scan, if_neg (by decide), root120Candidates17_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail18
    (if 3 ≤ root120Candidates17.length then search edge 3 [677, 5393] root120Candidates17 6070 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates18 : List Nat := [15773]
set_option maxRecDepth 8192 in
theorem root120Candidates18_checked : candidates edge 4 677 5399 root120Tail19 winner.bound = root120Candidates18 := by
  unfold candidates
  have row : edge 5399 = (SortedNatTree.contains FastRows.row5399) := by
    change Data.row5399.contains = (SortedNatTree.contains FastRows.row5399)
    exact FastRows.row5399_checked.symm
  rw [row]
  decide
theorem root120Step18 : scan edge (search edge 3) 4 [677] 677 root120Tail18 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail19 winner := by
  rw [root120Tail18, scan, if_neg (by decide), root120Candidates18_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail19
    (if 3 ≤ root120Candidates18.length then search edge 3 [677, 5399] root120Candidates18 6076 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates19 : List Nat := [7517, 15773]
set_option maxRecDepth 8192 in
theorem root120Candidates19_checked : candidates edge 4 677 5519 root120Tail20 winner.bound = root120Candidates19 := by
  unfold candidates
  have row : edge 5519 = (SortedNatTree.contains FastRows.row5519) := by
    change Data.row5519.contains = (SortedNatTree.contains FastRows.row5519)
    exact FastRows.row5519_checked.symm
  rw [row]
  decide
theorem root120Step19 : scan edge (search edge 3) 4 [677] 677 root120Tail19 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail20 winner := by
  rw [root120Tail19, scan, if_neg (by decide), root120Candidates19_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail20
    (if 3 ≤ root120Candidates19.length then search edge 3 [677, 5519] root120Candidates19 6196 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates20 : List Nat := [7949]
set_option maxRecDepth 8192 in
theorem root120Candidates20_checked : candidates edge 4 677 5807 root120Tail21 winner.bound = root120Candidates20 := by
  unfold candidates
  have row : edge 5807 = (SortedNatTree.contains FastRows.row5807) := by
    change Data.row5807.contains = (SortedNatTree.contains FastRows.row5807)
    exact FastRows.row5807_checked.symm
  rw [row]
  decide
theorem root120Step20 : scan edge (search edge 3) 4 [677] 677 root120Tail20 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail21 winner := by
  rw [root120Tail20, scan, if_neg (by decide), root120Candidates20_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail21
    (if 3 ≤ root120Candidates20.length then search edge 3 [677, 5807] root120Candidates20 6484 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates21 : List Nat := [5987]
set_option maxRecDepth 8192 in
theorem root120Candidates21_checked : candidates edge 4 677 5861 root120Tail22 winner.bound = root120Candidates21 := by
  unfold candidates
  have row : edge 5861 = (SortedNatTree.contains FastRows.row5861) := by
    change Data.row5861.contains = (SortedNatTree.contains FastRows.row5861)
    exact FastRows.row5861_checked.symm
  rw [row]
  decide
theorem root120Step21 : scan edge (search edge 3) 4 [677] 677 root120Tail21 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail22 winner := by
  rw [root120Tail21, scan, if_neg (by decide), root120Candidates21_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail22
    (if 3 ≤ root120Candidates21.length then search edge 3 [677, 5861] root120Candidates21 6538 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates22 : List Nat := [6491, 14321]
set_option maxRecDepth 8192 in
theorem root120Candidates22_checked : candidates edge 4 677 5987 root120Tail23 winner.bound = root120Candidates22 := by
  unfold candidates
  have row : edge 5987 = (SortedNatTree.contains FastRows.row5987) := by
    change Data.row5987.contains = (SortedNatTree.contains FastRows.row5987)
    exact FastRows.row5987_checked.symm
  rw [row]
  decide
theorem root120Step22 : scan edge (search edge 3) 4 [677] 677 root120Tail22 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail23 winner := by
  rw [root120Tail22, scan, if_neg (by decide), root120Candidates22_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail23
    (if 3 ≤ root120Candidates22.length then search edge 3 [677, 5987] root120Candidates22 6664 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates23 : List Nat := []
set_option maxRecDepth 8192 in
theorem root120Candidates23_checked : candidates edge 4 677 6101 root120Tail24 winner.bound = root120Candidates23 := by
  unfold candidates
  have row : edge 6101 = (SortedNatTree.contains FastRows.row6101) := by
    change Data.row6101.contains = (SortedNatTree.contains FastRows.row6101)
    exact FastRows.row6101_checked.symm
  rw [row]
  decide
theorem root120Step23 : scan edge (search edge 3) 4 [677] 677 root120Tail23 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail24 winner := by
  rw [root120Tail23, scan, if_neg (by decide), root120Candidates23_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail24
    (if 3 ≤ root120Candidates23.length then search edge 3 [677, 6101] root120Candidates23 6778 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates24 : List Nat := []
set_option maxRecDepth 8192 in
theorem root120Candidates24_checked : candidates edge 4 677 6173 root120Tail25 winner.bound = root120Candidates24 := by
  unfold candidates
  have row : edge 6173 = (SortedNatTree.contains FastRows.row6173) := by
    change Data.row6173.contains = (SortedNatTree.contains FastRows.row6173)
    exact FastRows.row6173_checked.symm
  rw [row]
  decide
theorem root120Step24 : scan edge (search edge 3) 4 [677] 677 root120Tail24 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail25 winner := by
  rw [root120Tail24, scan, if_neg (by decide), root120Candidates24_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail25
    (if 3 ≤ root120Candidates24.length then search edge 3 [677, 6173] root120Candidates24 6850 winner else winner) = _
  rw [if_neg (by decide)]
def root120Candidates25 : List Nat := []
set_option maxRecDepth 8192 in
theorem root120Candidates25_checked : candidates edge 4 677 6257 root120Tail26 winner.bound = root120Candidates25 := by
  unfold candidates
  have row : edge 6257 = (SortedNatTree.contains FastRows.row6257) := by
    change Data.row6257.contains = (SortedNatTree.contains FastRows.row6257)
    exact FastRows.row6257_checked.symm
  rw [row]
  decide
theorem root120Step25 : scan edge (search edge 3) 4 [677] 677 root120Tail25 winner =
    scan edge (search edge 3) 4 [677] 677 root120Tail26 winner := by
  rw [root120Tail25, scan, if_neg (by decide), root120Candidates25_checked]
  change scan edge (search edge 3) 4 [677] 677 root120Tail26
    (if 3 ≤ root120Candidates25.length then search edge 3 [677, 6257] root120Candidates25 6934 winner else winner) = _
  rw [if_neg (by decide)]
theorem root120Scan26 : scan edge (search edge 3) 4 [677] 677 root120Tail26 winner = winner := by rw [root120Tail26, scan, if_pos (by decide)]
theorem root120Scan25 : scan edge (search edge 3) 4 [677] 677 root120Tail25 winner = winner :=
  (root120Step25).trans root120Scan26
theorem root120Scan24 : scan edge (search edge 3) 4 [677] 677 root120Tail24 winner = winner :=
  (root120Step24).trans root120Scan25
theorem root120Scan23 : scan edge (search edge 3) 4 [677] 677 root120Tail23 winner = winner :=
  (root120Step23).trans root120Scan24
theorem root120Scan22 : scan edge (search edge 3) 4 [677] 677 root120Tail22 winner = winner :=
  (root120Step22).trans root120Scan23
theorem root120Scan21 : scan edge (search edge 3) 4 [677] 677 root120Tail21 winner = winner :=
  (root120Step21).trans root120Scan22
theorem root120Scan20 : scan edge (search edge 3) 4 [677] 677 root120Tail20 winner = winner :=
  (root120Step20).trans root120Scan21
theorem root120Scan19 : scan edge (search edge 3) 4 [677] 677 root120Tail19 winner = winner :=
  (root120Step19).trans root120Scan20
theorem root120Scan18 : scan edge (search edge 3) 4 [677] 677 root120Tail18 winner = winner :=
  (root120Step18).trans root120Scan19
theorem root120Scan17 : scan edge (search edge 3) 4 [677] 677 root120Tail17 winner = winner :=
  (root120Step17).trans root120Scan18
theorem root120Scan16 : scan edge (search edge 3) 4 [677] 677 root120Tail16 winner = winner :=
  (root120Step16).trans root120Scan17
theorem root120Scan15 : scan edge (search edge 3) 4 [677] 677 root120Tail15 winner = winner :=
  (root120Step15).trans root120Scan16
theorem root120Scan14 : scan edge (search edge 3) 4 [677] 677 root120Tail14 winner = winner :=
  (root120Step14).trans root120Scan15
theorem root120Scan13 : scan edge (search edge 3) 4 [677] 677 root120Tail13 winner = winner :=
  (root120Step13).trans root120Scan14
theorem root120Scan12 : scan edge (search edge 3) 4 [677] 677 root120Tail12 winner = winner :=
  (root120Step12).trans root120Scan13
theorem root120Scan11 : scan edge (search edge 3) 4 [677] 677 root120Tail11 winner = winner :=
  (root120Step11).trans root120Scan12
theorem root120Scan10 : scan edge (search edge 3) 4 [677] 677 root120Tail10 winner = winner :=
  (root120Step10).trans root120Scan11
theorem root120Scan9 : scan edge (search edge 3) 4 [677] 677 root120Tail9 winner = winner :=
  (root120Step9).trans root120Scan10
theorem root120Scan8 : scan edge (search edge 3) 4 [677] 677 root120Tail8 winner = winner :=
  (root120Step8).trans root120Scan9
theorem root120Scan7 : scan edge (search edge 3) 4 [677] 677 root120Tail7 winner = winner :=
  (root120Step7).trans root120Scan8
theorem root120Scan6 : scan edge (search edge 3) 4 [677] 677 root120Tail6 winner = winner :=
  (root120Step6).trans root120Scan7
theorem root120Scan5 : scan edge (search edge 3) 4 [677] 677 root120Tail5 winner = winner :=
  (root120Step5).trans root120Scan6
theorem root120Scan4 : scan edge (search edge 3) 4 [677] 677 root120Tail4 winner = winner :=
  (root120Step4).trans root120Scan5
theorem root120Scan3 : scan edge (search edge 3) 4 [677] 677 root120Tail3 winner = winner :=
  (root120Step3).trans root120Scan4
theorem root120Scan2 : scan edge (search edge 3) 4 [677] 677 root120Tail2 winner = winner :=
  (root120Step2).trans root120Scan3
theorem root120Scan1 : scan edge (search edge 3) 4 [677] 677 root120Tail1 winner = winner :=
  (root120Step1).trans root120Scan2
theorem root120Scan0 : scan edge (search edge 3) 4 [677] 677 root120Tail0 winner = winner :=
  (root120Step0).trans root120Scan1
set_option maxRecDepth 4096 in
theorem root120Branch : search edge 4 [677] root120Candidates 677 winner = winner := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [677] 677 root120Tail0 winner = winner
  exact root120Scan0
end ProjectEulerSolutions.PrimePairLarge.Search
