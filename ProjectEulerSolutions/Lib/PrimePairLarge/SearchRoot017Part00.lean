import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot017Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root17Candidates0 : List Nat := [1163, 1583, 7001, 11789, 13907, 21377]
set_option maxRecDepth 8192 in
theorem root17Candidates0_checked : candidates edge 4 71 233 root17Tail1 winner.bound = root17Candidates0 := by
  unfold candidates
  have row : edge 233 = (SortedNatTree.contains FastRows.row233) := by
    change Data.row233.contains = (SortedNatTree.contains FastRows.row233)
    exact FastRows.row233_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child0 : search edge 3 [71, 233] root17Candidates0 304 winner = winner := by decide
theorem root17Step0 : scan edge (search edge 3) 4 [71] 71 root17Tail0 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail1 winner := by
  rw [root17Tail0, scan, if_neg (by decide), root17Candidates0_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail1
    (if 3 ≤ root17Candidates0.length then search edge 3 [71, 233] root17Candidates0 304 winner else winner) = _
  rw [if_pos (by decide), root17Child0]
def root17Candidates1 : List Nat := [263, 3833, 8423]
set_option maxRecDepth 8192 in
theorem root17Candidates1_checked : candidates edge 4 71 257 root17Tail2 winner.bound = root17Candidates1 := by
  unfold candidates
  have row : edge 257 = (SortedNatTree.contains FastRows.row257) := by
    change Data.row257.contains = (SortedNatTree.contains FastRows.row257)
    exact FastRows.row257_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child1 : search edge 3 [71, 257] root17Candidates1 328 winner = winner := by decide
theorem root17Step1 : scan edge (search edge 3) 4 [71] 71 root17Tail1 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail2 winner := by
  rw [root17Tail1, scan, if_neg (by decide), root17Candidates1_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail2
    (if 3 ≤ root17Candidates1.length then search edge 3 [71, 257] root17Candidates1 328 winner else winner) = _
  rw [if_pos (by decide), root17Child1]
def root17Candidates2 : List Nat := [443, 821, 1049, 2477, 8087]
set_option maxRecDepth 8192 in
theorem root17Candidates2_checked : candidates edge 4 71 263 root17Tail3 winner.bound = root17Candidates2 := by
  unfold candidates
  have row : edge 263 = (SortedNatTree.contains FastRows.row263) := by
    change Data.row263.contains = (SortedNatTree.contains FastRows.row263)
    exact FastRows.row263_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child2 : search edge 3 [71, 263] root17Candidates2 334 winner = winner := by decide
theorem root17Step2 : scan edge (search edge 3) 4 [71] 71 root17Tail2 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail3 winner := by
  rw [root17Tail2, scan, if_neg (by decide), root17Candidates2_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail3
    (if 3 ≤ root17Candidates2.length then search edge 3 [71, 263] root17Candidates2 334 winner else winner) = _
  rw [if_pos (by decide), root17Child2]
def root17Candidates3 : List Nat := [1709, 8999, 9749, 12011]
set_option maxRecDepth 8192 in
theorem root17Candidates3_checked : candidates edge 4 71 317 root17Tail4 winner.bound = root17Candidates3 := by
  unfold candidates
  have row : edge 317 = (SortedNatTree.contains FastRows.row317) := by
    change Data.row317.contains = (SortedNatTree.contains FastRows.row317)
    exact FastRows.row317_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child3 : search edge 3 [71, 317] root17Candidates3 388 winner = winner := by decide
theorem root17Step3 : scan edge (search edge 3) 4 [71] 71 root17Tail3 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail4 winner := by
  rw [root17Tail3, scan, if_neg (by decide), root17Candidates3_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail4
    (if 3 ≤ root17Candidates3.length then search edge 3 [71, 317] root17Candidates3 388 winner else winner) = _
  rw [if_pos (by decide), root17Child3]
def root17Candidates4 : List Nat := [947, 971, 8087]
set_option maxRecDepth 8192 in
theorem root17Candidates4_checked : candidates edge 4 71 389 root17Tail5 winner.bound = root17Candidates4 := by
  unfold candidates
  have row : edge 389 = (SortedNatTree.contains FastRows.row389) := by
    change Data.row389.contains = (SortedNatTree.contains FastRows.row389)
    exact FastRows.row389_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child4 : search edge 3 [71, 389] root17Candidates4 460 winner = winner := by decide
theorem root17Step4 : scan edge (search edge 3) 4 [71] 71 root17Tail4 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail5 winner := by
  rw [root17Tail4, scan, if_neg (by decide), root17Candidates4_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail5
    (if 3 ≤ root17Candidates4.length then search edge 3 [71, 389] root17Candidates4 460 winner else winner) = _
  rw [if_pos (by decide), root17Child4]
def root17Candidates5 : List Nat := [3833, 4673, 6389, 17729, 25121]
set_option maxRecDepth 8192 in
theorem root17Candidates5_checked : candidates edge 4 71 443 root17Tail6 winner.bound = root17Candidates5 := by
  unfold candidates
  have row : edge 443 = (SortedNatTree.contains FastRows.row443) := by
    change Data.row443.contains = (SortedNatTree.contains FastRows.row443)
    exact FastRows.row443_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child5 : search edge 3 [71, 443] root17Candidates5 514 winner = winner := by decide
theorem root17Step5 : scan edge (search edge 3) 4 [71] 71 root17Tail5 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail6 winner := by
  rw [root17Tail5, scan, if_neg (by decide), root17Candidates5_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail6
    (if 3 ≤ root17Candidates5.length then search edge 3 [71, 443] root17Candidates5 514 winner else winner) = _
  rw [if_pos (by decide), root17Child5]
def root17Candidates6 : List Nat := [947, 3833, 6869]
set_option maxRecDepth 8192 in
theorem root17Candidates6_checked : candidates edge 4 71 719 root17Tail7 winner.bound = root17Candidates6 := by
  unfold candidates
  have row : edge 719 = (SortedNatTree.contains FastRows.row719) := by
    change Data.row719.contains = (SortedNatTree.contains FastRows.row719)
    exact FastRows.row719_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child6 : search edge 3 [71, 719] root17Candidates6 790 winner = winner := by decide
theorem root17Step6 : scan edge (search edge 3) 4 [71] 71 root17Tail6 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail7 winner := by
  rw [root17Tail6, scan, if_neg (by decide), root17Candidates6_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail7
    (if 3 ≤ root17Candidates6.length then search edge 3 [71, 719] root17Candidates6 790 winner else winner) = _
  rw [if_pos (by decide), root17Child6]
def root17Candidates7 : List Nat := [971, 1163, 5849, 14957, 22961]
set_option maxRecDepth 8192 in
theorem root17Candidates7_checked : candidates edge 4 71 821 root17Tail8 winner.bound = root17Candidates7 := by
  unfold candidates
  have row : edge 821 = (SortedNatTree.contains FastRows.row821) := by
    change Data.row821.contains = (SortedNatTree.contains FastRows.row821)
    exact FastRows.row821_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child7 : search edge 3 [71, 821] root17Candidates7 892 winner = winner := by decide
theorem root17Step7 : scan edge (search edge 3) 4 [71] 71 root17Tail7 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail8 winner := by
  rw [root17Tail7, scan, if_neg (by decide), root17Candidates7_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail8
    (if 3 ≤ root17Candidates7.length then search edge 3 [71, 821] root17Candidates7 892 winner else winner) = _
  rw [if_pos (by decide), root17Child7]
def root17Candidates8 : List Nat := [1427, 5417, 5651, 7109, 8087, 8423, 11789, 15971, 22691]
set_option maxRecDepth 8192 in
theorem root17Candidates8_checked : candidates edge 4 71 887 root17Tail9 winner.bound = root17Candidates8 := by
  unfold candidates
  have row : edge 887 = (SortedNatTree.contains FastRows.row887) := by
    change Data.row887.contains = (SortedNatTree.contains FastRows.row887)
    exact FastRows.row887_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child8 : search edge 3 [71, 887] root17Candidates8 958 winner = winner := by decide
theorem root17Step8 : scan edge (search edge 3) 4 [71] 71 root17Tail8 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail9 winner := by
  rw [root17Tail8, scan, if_neg (by decide), root17Candidates8_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail9
    (if 3 ≤ root17Candidates8.length then search edge 3 [71, 887] root17Candidates8 958 winner else winner) = _
  rw [if_pos (by decide), root17Child8]
def root17Candidates9 : List Nat := [3491, 8387, 9203, 9419, 17729]
set_option maxRecDepth 8192 in
theorem root17Candidates9_checked : candidates edge 4 71 947 root17Tail10 winner.bound = root17Candidates9 := by
  unfold candidates
  have row : edge 947 = (SortedNatTree.contains FastRows.row947) := by
    change Data.row947.contains = (SortedNatTree.contains FastRows.row947)
    exact FastRows.row947_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child9 : search edge 3 [71, 947] root17Candidates9 1018 winner = winner := by decide
theorem root17Step9 : scan edge (search edge 3) 4 [71] 71 root17Tail9 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail10 winner := by
  rw [root17Tail9, scan, if_neg (by decide), root17Candidates9_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail10
    (if 3 ≤ root17Candidates9.length then search edge 3 [71, 947] root17Candidates9 1018 winner else winner) = _
  rw [if_pos (by decide), root17Child9]
def root17Candidates10 : List Nat := [3863, 7919, 8999, 16871]
set_option maxRecDepth 8192 in
theorem root17Candidates10_checked : candidates edge 4 71 971 root17Tail11 winner.bound = root17Candidates10 := by
  unfold candidates
  have row : edge 971 = (SortedNatTree.contains FastRows.row971) := by
    change Data.row971.contains = (SortedNatTree.contains FastRows.row971)
    exact FastRows.row971_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child10 : search edge 3 [71, 971] root17Candidates10 1042 winner = winner := by decide
theorem root17Step10 : scan edge (search edge 3) 4 [71] 71 root17Tail10 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail11 winner := by
  rw [root17Tail10, scan, if_neg (by decide), root17Candidates10_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail11
    (if 3 ≤ root17Candidates10.length then search edge 3 [71, 971] root17Candidates10 1042 winner else winner) = _
  rw [if_pos (by decide), root17Child10]
def root17Candidates11 : List Nat := [1181, 7229, 7841, 11789, 12011]
set_option maxRecDepth 8192 in
theorem root17Candidates11_checked : candidates edge 4 71 1049 root17Tail12 winner.bound = root17Candidates11 := by
  unfold candidates
  have row : edge 1049 = (SortedNatTree.contains FastRows.row1049) := by
    change Data.row1049.contains = (SortedNatTree.contains FastRows.row1049)
    exact FastRows.row1049_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child11 : search edge 3 [71, 1049] root17Candidates11 1120 winner = winner := by decide
theorem root17Step11 : scan edge (search edge 3) 4 [71] 71 root17Tail11 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail12 winner := by
  rw [root17Tail11, scan, if_neg (by decide), root17Candidates11_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail12
    (if 3 ≤ root17Candidates11.length then search edge 3 [71, 1049] root17Candidates11 1120 winner else winner) = _
  rw [if_pos (by decide), root17Child11]
def root17Candidates12 : List Nat := [2417, 4673, 5639, 5849, 7919, 10847]
set_option maxRecDepth 8192 in
theorem root17Candidates12_checked : candidates edge 4 71 1163 root17Tail13 winner.bound = root17Candidates12 := by
  unfold candidates
  have row : edge 1163 = (SortedNatTree.contains FastRows.row1163) := by
    change Data.row1163.contains = (SortedNatTree.contains FastRows.row1163)
    exact FastRows.row1163_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child12 : search edge 3 [71, 1163] root17Candidates12 1234 winner = winner := by decide
theorem root17Step12 : scan edge (search edge 3) 4 [71] 71 root17Tail12 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail13 winner := by
  rw [root17Tail12, scan, if_neg (by decide), root17Candidates12_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail13
    (if 3 ≤ root17Candidates12.length then search edge 3 [71, 1163] root17Candidates12 1234 winner else winner) = _
  rw [if_pos (by decide), root17Child12]
def root17Candidates13 : List Nat := [1427, 2687]
set_option maxRecDepth 8192 in
theorem root17Candidates13_checked : candidates edge 4 71 1181 root17Tail14 winner.bound = root17Candidates13 := by
  unfold candidates
  have row : edge 1181 = (SortedNatTree.contains FastRows.row1181) := by
    change Data.row1181.contains = (SortedNatTree.contains FastRows.row1181)
    exact FastRows.row1181_checked.symm
  rw [row]
  decide
theorem root17Step13 : scan edge (search edge 3) 4 [71] 71 root17Tail13 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail14 winner := by
  rw [root17Tail13, scan, if_neg (by decide), root17Candidates13_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail14
    (if 3 ≤ root17Candidates13.length then search edge 3 [71, 1181] root17Candidates13 1252 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates14 : List Nat := [13721, 14753, 22469]
set_option maxRecDepth 8192 in
theorem root17Candidates14_checked : candidates edge 4 71 1427 root17Tail15 winner.bound = root17Candidates14 := by
  unfold candidates
  have row : edge 1427 = (SortedNatTree.contains FastRows.row1427) := by
    change Data.row1427.contains = (SortedNatTree.contains FastRows.row1427)
    exact FastRows.row1427_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child14 : search edge 3 [71, 1427] root17Candidates14 1498 winner = winner := by decide
theorem root17Step14 : scan edge (search edge 3) 4 [71] 71 root17Tail14 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail15 winner := by
  rw [root17Tail14, scan, if_neg (by decide), root17Candidates14_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail15
    (if 3 ≤ root17Candidates14.length then search edge 3 [71, 1427] root17Candidates14 1498 winner else winner) = _
  rw [if_pos (by decide), root17Child14]
def root17Candidates15 : List Nat := [6563, 7589, 19403]
set_option maxRecDepth 8192 in
theorem root17Candidates15_checked : candidates edge 4 71 1499 root17Tail16 winner.bound = root17Candidates15 := by
  unfold candidates
  have row : edge 1499 = (SortedNatTree.contains FastRows.row1499) := by
    change Data.row1499.contains = (SortedNatTree.contains FastRows.row1499)
    exact FastRows.row1499_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child15 : search edge 3 [71, 1499] root17Candidates15 1570 winner = winner := by decide
theorem root17Step15 : scan edge (search edge 3) 4 [71] 71 root17Tail15 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail16 winner := by
  rw [root17Tail15, scan, if_neg (by decide), root17Candidates15_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail16
    (if 3 ≤ root17Candidates15.length then search edge 3 [71, 1499] root17Candidates15 1570 winner else winner) = _
  rw [if_pos (by decide), root17Child15]
def root17Candidates16 : List Nat := [1973, 3347, 5879, 6869]
set_option maxRecDepth 8192 in
theorem root17Candidates16_checked : candidates edge 4 71 1583 root17Tail17 winner.bound = root17Candidates16 := by
  unfold candidates
  have row : edge 1583 = (SortedNatTree.contains FastRows.row1583) := by
    change Data.row1583.contains = (SortedNatTree.contains FastRows.row1583)
    exact FastRows.row1583_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child16 : search edge 3 [71, 1583] root17Candidates16 1654 winner = winner := by decide
theorem root17Step16 : scan edge (search edge 3) 4 [71] 71 root17Tail16 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail17 winner := by
  rw [root17Tail16, scan, if_neg (by decide), root17Candidates16_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail17
    (if 3 ≤ root17Candidates16.length then search edge 3 [71, 1583] root17Candidates16 1654 winner else winner) = _
  rw [if_pos (by decide), root17Child16]
def root17Candidates17 : List Nat := [6869, 7001, 15971, 22091]
set_option maxRecDepth 8192 in
theorem root17Candidates17_checked : candidates edge 4 71 1709 root17Tail18 winner.bound = root17Candidates17 := by
  unfold candidates
  have row : edge 1709 = (SortedNatTree.contains FastRows.row1709) := by
    change Data.row1709.contains = (SortedNatTree.contains FastRows.row1709)
    exact FastRows.row1709_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child17 : search edge 3 [71, 1709] root17Candidates17 1780 winner = winner := by decide
theorem root17Step17 : scan edge (search edge 3) 4 [71] 71 root17Tail17 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail18 winner := by
  rw [root17Tail17, scan, if_neg (by decide), root17Candidates17_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail18
    (if 3 ≤ root17Candidates17.length then search edge 3 [71, 1709] root17Candidates17 1780 winner else winner) = _
  rw [if_pos (by decide), root17Child17]
def root17Candidates18 : List Nat := []
set_option maxRecDepth 8192 in
theorem root17Candidates18_checked : candidates edge 4 71 1973 root17Tail19 winner.bound = root17Candidates18 := by
  unfold candidates
  have row : edge 1973 = (SortedNatTree.contains FastRows.row1973) := by
    change Data.row1973.contains = (SortedNatTree.contains FastRows.row1973)
    exact FastRows.row1973_checked.symm
  rw [row]
  decide
theorem root17Step18 : scan edge (search edge 3) 4 [71] 71 root17Tail18 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail19 winner := by
  rw [root17Tail18, scan, if_neg (by decide), root17Candidates18_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail19
    (if 3 ≤ root17Candidates18.length then search edge 3 [71, 1973] root17Candidates18 2044 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates19 : List Nat := [5153, 6389, 6563, 7103, 8747, 13553]
set_option maxRecDepth 8192 in
theorem root17Candidates19_checked : candidates edge 4 71 2417 root17Tail20 winner.bound = root17Candidates19 := by
  unfold candidates
  have row : edge 2417 = (SortedNatTree.contains FastRows.row2417) := by
    change Data.row2417.contains = (SortedNatTree.contains FastRows.row2417)
    exact FastRows.row2417_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child19 : search edge 3 [71, 2417] root17Candidates19 2488 winner = winner := by decide
theorem root17Step19 : scan edge (search edge 3) 4 [71] 71 root17Tail19 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail20 winner := by
  rw [root17Tail19, scan, if_neg (by decide), root17Candidates19_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail20
    (if 3 ≤ root17Candidates19.length then search edge 3 [71, 2417] root17Candidates19 2488 winner else winner) = _
  rw [if_pos (by decide), root17Child19]
def root17Candidates20 : List Nat := [11549, 15737, 16871]
set_option maxRecDepth 8192 in
theorem root17Candidates20_checked : candidates edge 4 71 2477 root17Tail21 winner.bound = root17Candidates20 := by
  unfold candidates
  have row : edge 2477 = (SortedNatTree.contains FastRows.row2477) := by
    change Data.row2477.contains = (SortedNatTree.contains FastRows.row2477)
    exact FastRows.row2477_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child20 : search edge 3 [71, 2477] root17Candidates20 2548 winner = winner := by decide
theorem root17Step20 : scan edge (search edge 3) 4 [71] 71 root17Tail20 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail21 winner := by
  rw [root17Tail20, scan, if_neg (by decide), root17Candidates20_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail21
    (if 3 ≤ root17Candidates20.length then search edge 3 [71, 2477] root17Candidates20 2548 winner else winner) = _
  rw [if_pos (by decide), root17Child20]
def root17Candidates21 : List Nat := [5237, 11087, 15971, 21767]
set_option maxRecDepth 8192 in
theorem root17Candidates21_checked : candidates edge 4 71 2687 root17Tail22 winner.bound = root17Candidates21 := by
  unfold candidates
  have row : edge 2687 = (SortedNatTree.contains FastRows.row2687) := by
    change Data.row2687.contains = (SortedNatTree.contains FastRows.row2687)
    exact FastRows.row2687_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child21 : search edge 3 [71, 2687] root17Candidates21 2758 winner = winner := by decide
theorem root17Step21 : scan edge (search edge 3) 4 [71] 71 root17Tail21 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail22 winner := by
  rw [root17Tail21, scan, if_neg (by decide), root17Candidates21_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail22
    (if 3 ≤ root17Candidates21.length then search edge 3 [71, 2687] root17Candidates21 2758 winner else winner) = _
  rw [if_pos (by decide), root17Child21]
def root17Candidates22 : List Nat := [5879, 9203]
set_option maxRecDepth 8192 in
theorem root17Candidates22_checked : candidates edge 4 71 2819 root17Tail23 winner.bound = root17Candidates22 := by
  unfold candidates
  have row : edge 2819 = (SortedNatTree.contains FastRows.row2819) := by
    change Data.row2819.contains = (SortedNatTree.contains FastRows.row2819)
    exact FastRows.row2819_checked.symm
  rw [row]
  decide
theorem root17Step22 : scan edge (search edge 3) 4 [71] 71 root17Tail22 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail23 winner := by
  rw [root17Tail22, scan, if_neg (by decide), root17Candidates22_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail23
    (if 3 ≤ root17Candidates22.length then search edge 3 [71, 2819] root17Candidates22 2890 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates23 : List Nat := [5153, 7001, 9419, 13721]
set_option maxRecDepth 8192 in
theorem root17Candidates23_checked : candidates edge 4 71 2837 root17Tail24 winner.bound = root17Candidates23 := by
  unfold candidates
  have row : edge 2837 = (SortedNatTree.contains FastRows.row2837) := by
    change Data.row2837.contains = (SortedNatTree.contains FastRows.row2837)
    exact FastRows.row2837_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child23 : search edge 3 [71, 2837] root17Candidates23 2908 winner = winner := by decide
theorem root17Step23 : scan edge (search edge 3) 4 [71] 71 root17Tail23 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail24 winner := by
  rw [root17Tail23, scan, if_neg (by decide), root17Candidates23_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail24
    (if 3 ≤ root17Candidates23.length then search edge 3 [71, 2837] root17Candidates23 2908 winner else winner) = _
  rw [if_pos (by decide), root17Child23]
def root17Candidates24 : List Nat := [5153, 9011, 9419, 9689, 21767, 22013]
set_option maxRecDepth 8192 in
theorem root17Candidates24_checked : candidates edge 4 71 2909 root17Tail25 winner.bound = root17Candidates24 := by
  unfold candidates
  have row : edge 2909 = (SortedNatTree.contains FastRows.row2909) := by
    change Data.row2909.contains = (SortedNatTree.contains FastRows.row2909)
    exact FastRows.row2909_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child24 : search edge 3 [71, 2909] root17Candidates24 2980 winner = winner := by decide
theorem root17Step24 : scan edge (search edge 3) 4 [71] 71 root17Tail24 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail25 winner := by
  rw [root17Tail24, scan, if_neg (by decide), root17Candidates24_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail25
    (if 3 ≤ root17Candidates24.length then search edge 3 [71, 2909] root17Candidates24 2980 winner else winner) = _
  rw [if_pos (by decide), root17Child24]
def root17Candidates25 : List Nat := [4673, 8999, 12041, 15971, 21377]
set_option maxRecDepth 8192 in
theorem root17Candidates25_checked : candidates edge 4 71 3347 root17Tail26 winner.bound = root17Candidates25 := by
  unfold candidates
  have row : edge 3347 = (SortedNatTree.contains FastRows.row3347) := by
    change Data.row3347.contains = (SortedNatTree.contains FastRows.row3347)
    exact FastRows.row3347_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child25 : search edge 3 [71, 3347] root17Candidates25 3418 winner = winner := by decide
theorem root17Step25 : scan edge (search edge 3) 4 [71] 71 root17Tail25 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail26 winner := by
  rw [root17Tail25, scan, if_neg (by decide), root17Candidates25_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail26
    (if 3 ≤ root17Candidates25.length then search edge 3 [71, 3347] root17Candidates25 3418 winner else winner) = _
  rw [if_pos (by decide), root17Child25]
def root17Candidates26 : List Nat := []
set_option maxRecDepth 8192 in
theorem root17Candidates26_checked : candidates edge 4 71 3491 root17Tail27 winner.bound = root17Candidates26 := by
  unfold candidates
  have row : edge 3491 = (SortedNatTree.contains FastRows.row3491) := by
    change Data.row3491.contains = (SortedNatTree.contains FastRows.row3491)
    exact FastRows.row3491_checked.symm
  rw [row]
  decide
theorem root17Step26 : scan edge (search edge 3) 4 [71] 71 root17Tail26 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail27 winner := by
  rw [root17Tail26, scan, if_neg (by decide), root17Candidates26_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail27
    (if 3 ≤ root17Candidates26.length then search edge 3 [71, 3491] root17Candidates26 3562 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates27 : List Nat := [5657, 7841, 14621, 14753]
set_option maxRecDepth 8192 in
theorem root17Candidates27_checked : candidates edge 4 71 3833 root17Tail28 winner.bound = root17Candidates27 := by
  unfold candidates
  have row : edge 3833 = (SortedNatTree.contains FastRows.row3833) := by
    change Data.row3833.contains = (SortedNatTree.contains FastRows.row3833)
    exact FastRows.row3833_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child27 : search edge 3 [71, 3833] root17Candidates27 3904 winner = winner := by decide
theorem root17Step27 : scan edge (search edge 3) 4 [71] 71 root17Tail27 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail28 winner := by
  rw [root17Tail27, scan, if_neg (by decide), root17Candidates27_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail28
    (if 3 ≤ root17Candidates27.length then search edge 3 [71, 3833] root17Candidates27 3904 winner else winner) = _
  rw [if_pos (by decide), root17Child27]
def root17Candidates28 : List Nat := [9203, 11789, 12041]
set_option maxRecDepth 8192 in
theorem root17Candidates28_checked : candidates edge 4 71 3863 root17Tail29 winner.bound = root17Candidates28 := by
  unfold candidates
  have row : edge 3863 = (SortedNatTree.contains FastRows.row3863) := by
    change Data.row3863.contains = (SortedNatTree.contains FastRows.row3863)
    exact FastRows.row3863_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child28 : search edge 3 [71, 3863] root17Candidates28 3934 winner = winner := by decide
theorem root17Step28 : scan edge (search edge 3) 4 [71] 71 root17Tail28 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail29 winner := by
  rw [root17Tail28, scan, if_neg (by decide), root17Candidates28_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail29
    (if 3 ≤ root17Candidates28.length then search edge 3 [71, 3863] root17Candidates28 3934 winner else winner) = _
  rw [if_pos (by decide), root17Child28]
def root17Candidates29 : List Nat := [5087, 5849, 6869, 8087, 8387, 9689, 17729]
set_option maxRecDepth 8192 in
theorem root17Candidates29_checked : candidates edge 4 71 4517 root17Tail30 winner.bound = root17Candidates29 := by
  unfold candidates
  have row : edge 4517 = (SortedNatTree.contains FastRows.row4517) := by
    change Data.row4517.contains = (SortedNatTree.contains FastRows.row4517)
    exact FastRows.row4517_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root17Child29 : search edge 3 [71, 4517] root17Candidates29 4588 winner = winner := by decide
theorem root17Step29 : scan edge (search edge 3) 4 [71] 71 root17Tail29 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail30 winner := by
  rw [root17Tail29, scan, if_neg (by decide), root17Candidates29_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail30
    (if 3 ≤ root17Candidates29.length then search edge 3 [71, 4517] root17Candidates29 4588 winner else winner) = _
  rw [if_pos (by decide), root17Child29]
def root17Candidates30 : List Nat := [14957]
set_option maxRecDepth 8192 in
theorem root17Candidates30_checked : candidates edge 4 71 4673 root17Tail31 winner.bound = root17Candidates30 := by
  unfold candidates
  have row : edge 4673 = (SortedNatTree.contains FastRows.row4673) := by
    change Data.row4673.contains = (SortedNatTree.contains FastRows.row4673)
    exact FastRows.row4673_checked.symm
  rw [row]
  decide
theorem root17Step30 : scan edge (search edge 3) 4 [71] 71 root17Tail30 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail31 winner := by
  rw [root17Tail30, scan, if_neg (by decide), root17Candidates30_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail31
    (if 3 ≤ root17Candidates30.length then search edge 3 [71, 4673] root17Candidates30 4744 winner else winner) = _
  rw [if_neg (by decide)]
def root17Candidates31 : List Nat := [11789]
set_option maxRecDepth 8192 in
theorem root17Candidates31_checked : candidates edge 4 71 5087 root17Tail32 winner.bound = root17Candidates31 := by
  unfold candidates
  have row : edge 5087 = (SortedNatTree.contains FastRows.row5087) := by
    change Data.row5087.contains = (SortedNatTree.contains FastRows.row5087)
    exact FastRows.row5087_checked.symm
  rw [row]
  decide
theorem root17Step31 : scan edge (search edge 3) 4 [71] 71 root17Tail31 winner =
    scan edge (search edge 3) 4 [71] 71 root17Tail32 winner := by
  rw [root17Tail31, scan, if_neg (by decide), root17Candidates31_checked]
  change scan edge (search edge 3) 4 [71] 71 root17Tail32
    (if 3 ≤ root17Candidates31.length then search edge 3 [71, 5087] root17Candidates31 5158 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
