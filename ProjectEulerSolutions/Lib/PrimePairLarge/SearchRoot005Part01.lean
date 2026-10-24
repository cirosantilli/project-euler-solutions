import ProjectEulerSolutions.Lib.PrimePairLarge.SearchRoot005Data
namespace ProjectEulerSolutions.PrimePairLarge.Search
open CliqueSearch
def root5Candidates32 : List Nat := [5737, 6961, 7741, 14767, 22153]
set_option maxRecDepth 8192 in
theorem root5Candidates32_checked : candidates edge 4 19 2677 root5Tail33 winner.bound = root5Candidates32 := by
  unfold candidates
  have row : edge 2677 = (SortedNatTree.contains FastRows.row2677) := by
    change Data.row2677.contains = (SortedNatTree.contains FastRows.row2677)
    exact FastRows.row2677_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child32 : search edge 3 [19, 2677] root5Candidates32 2696 winner = winner := by decide
theorem root5Step32 : scan edge (search edge 3) 4 [19] 19 root5Tail32 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail33 winner := by
  rw [root5Tail32, scan, if_neg (by decide), root5Candidates32_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail33
    (if 3 ≤ root5Candidates32.length then search edge 3 [19, 2677] root5Candidates32 2696 winner else winner) = _
  rw [if_pos (by decide), root5Child32]
def root5Candidates33 : List Nat := [6961, 8179, 10177, 15619]
set_option maxRecDepth 8192 in
theorem root5Candidates33_checked : candidates edge 4 19 2791 root5Tail34 winner.bound = root5Candidates33 := by
  unfold candidates
  have row : edge 2791 = (SortedNatTree.contains FastRows.row2791) := by
    change Data.row2791.contains = (SortedNatTree.contains FastRows.row2791)
    exact FastRows.row2791_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child33 : search edge 3 [19, 2791] root5Candidates33 2810 winner = winner := by decide
theorem root5Step33 : scan edge (search edge 3) 4 [19] 19 root5Tail33 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail34 winner := by
  rw [root5Tail33, scan, if_neg (by decide), root5Candidates33_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail34
    (if 3 ≤ root5Candidates33.length then search edge 3 [19, 2791] root5Candidates33 2810 winner else winner) = _
  rw [if_pos (by decide), root5Child33]
def root5Candidates34 : List Nat := [5659, 6961, 7699, 8623, 9049]
set_option maxRecDepth 8192 in
theorem root5Candidates34_checked : candidates edge 4 19 3727 root5Tail35 winner.bound = root5Candidates34 := by
  unfold candidates
  have row : edge 3727 = (SortedNatTree.contains FastRows.row3727) := by
    change Data.row3727.contains = (SortedNatTree.contains FastRows.row3727)
    exact FastRows.row3727_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child34 : search edge 3 [19, 3727] root5Candidates34 3746 winner = winner := by decide
theorem root5Step34 : scan edge (search edge 3) 4 [19] 19 root5Tail34 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail35 winner := by
  rw [root5Tail34, scan, if_neg (by decide), root5Candidates34_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail35
    (if 3 ≤ root5Candidates34.length then search edge 3 [19, 3727] root5Candidates34 3746 winner else winner) = _
  rw [if_pos (by decide), root5Child34]
def root5Candidates35 : List Nat := [5659, 6907, 8179]
set_option maxRecDepth 8192 in
theorem root5Candidates35_checked : candidates edge 4 19 3793 root5Tail36 winner.bound = root5Candidates35 := by
  unfold candidates
  have row : edge 3793 = (SortedNatTree.contains FastRows.row3793) := by
    change Data.row3793.contains = (SortedNatTree.contains FastRows.row3793)
    exact FastRows.row3793_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child35 : search edge 3 [19, 3793] root5Candidates35 3812 winner = winner := by decide
theorem root5Step35 : scan edge (search edge 3) 4 [19] 19 root5Tail35 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail36 winner := by
  rw [root5Tail35, scan, if_neg (by decide), root5Candidates35_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail36
    (if 3 ≤ root5Candidates35.length then search edge 3 [19, 3793] root5Candidates35 3812 winner else winner) = _
  rw [if_pos (by decide), root5Child35]
def root5Candidates36 : List Nat := [5077, 5659, 5737, 8929, 19273]
set_option maxRecDepth 8192 in
theorem root5Candidates36_checked : candidates edge 4 19 3847 root5Tail37 winner.bound = root5Candidates36 := by
  unfold candidates
  have row : edge 3847 = (SortedNatTree.contains FastRows.row3847) := by
    change Data.row3847.contains = (SortedNatTree.contains FastRows.row3847)
    exact FastRows.row3847_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child36 : search edge 3 [19, 3847] root5Candidates36 3866 winner = winner := by decide
theorem root5Step36 : scan edge (search edge 3) 4 [19] 19 root5Tail36 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail37 winner := by
  rw [root5Tail36, scan, if_neg (by decide), root5Candidates36_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail37
    (if 3 ≤ root5Candidates36.length then search edge 3 [19, 3847] root5Candidates36 3866 winner else winner) = _
  rw [if_pos (by decide), root5Child36]
def root5Candidates37 : List Nat := [12277, 14593, 16183]
set_option maxRecDepth 8192 in
theorem root5Candidates37_checked : candidates edge 4 19 3943 root5Tail38 winner.bound = root5Candidates37 := by
  unfold candidates
  have row : edge 3943 = (SortedNatTree.contains FastRows.row3943) := by
    change Data.row3943.contains = (SortedNatTree.contains FastRows.row3943)
    exact FastRows.row3943_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child37 : search edge 3 [19, 3943] root5Candidates37 3962 winner = winner := by decide
theorem root5Step37 : scan edge (search edge 3) 4 [19] 19 root5Tail37 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail38 winner := by
  rw [root5Tail37, scan, if_neg (by decide), root5Candidates37_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail38
    (if 3 ≤ root5Candidates37.length then search edge 3 [19, 3943] root5Candidates37 3962 winner else winner) = _
  rw [if_pos (by decide), root5Child37]
def root5Candidates38 : List Nat := [5077, 12583, 19927]
set_option maxRecDepth 8192 in
theorem root5Candidates38_checked : candidates edge 4 19 4057 root5Tail39 winner.bound = root5Candidates38 := by
  unfold candidates
  have row : edge 4057 = (SortedNatTree.contains FastRows.row4057) := by
    change Data.row4057.contains = (SortedNatTree.contains FastRows.row4057)
    exact FastRows.row4057_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child38 : search edge 3 [19, 4057] root5Candidates38 4076 winner = winner := by decide
theorem root5Step38 : scan edge (search edge 3) 4 [19] 19 root5Tail38 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail39 winner := by
  rw [root5Tail38, scan, if_neg (by decide), root5Candidates38_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail39
    (if 3 ≤ root5Candidates38.length then search edge 3 [19, 4057] root5Candidates38 4076 winner else winner) = _
  rw [if_pos (by decide), root5Child38]
def root5Candidates39 : List Nat := [6709, 7759, 12277, 12739, 18451]
set_option maxRecDepth 8192 in
theorem root5Candidates39_checked : candidates edge 4 19 4507 root5Tail40 winner.bound = root5Candidates39 := by
  unfold candidates
  have row : edge 4507 = (SortedNatTree.contains FastRows.row4507) := by
    change Data.row4507.contains = (SortedNatTree.contains FastRows.row4507)
    exact FastRows.row4507_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child39 : search edge 3 [19, 4507] root5Candidates39 4526 winner = winner := by decide
theorem root5Step39 : scan edge (search edge 3) 4 [19] 19 root5Tail39 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail40 winner := by
  rw [root5Tail39, scan, if_neg (by decide), root5Candidates39_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail40
    (if 3 ≤ root5Candidates39.length then search edge 3 [19, 4507] root5Candidates39 4526 winner else winner) = _
  rw [if_pos (by decide), root5Child39]
def root5Candidates40 : List Nat := [5869, 9721, 10159, 16333]
set_option maxRecDepth 8192 in
theorem root5Candidates40_checked : candidates edge 4 19 4723 root5Tail41 winner.bound = root5Candidates40 := by
  unfold candidates
  have row : edge 4723 = (SortedNatTree.contains FastRows.row4723) := by
    change Data.row4723.contains = (SortedNatTree.contains FastRows.row4723)
    exact FastRows.row4723_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child40 : search edge 3 [19, 4723] root5Candidates40 4742 winner = winner := by decide
theorem root5Step40 : scan edge (search edge 3) 4 [19] 19 root5Tail40 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail41 winner := by
  rw [root5Tail40, scan, if_neg (by decide), root5Candidates40_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail41
    (if 3 ≤ root5Candidates40.length then search edge 3 [19, 4723] root5Candidates40 4742 winner else winner) = _
  rw [if_pos (by decide), root5Child40]
def root5Candidates41 : List Nat := [5659, 7573, 12577, 18517]
set_option maxRecDepth 8192 in
theorem root5Candidates41_checked : candidates edge 4 19 4861 root5Tail42 winner.bound = root5Candidates41 := by
  unfold candidates
  have row : edge 4861 = (SortedNatTree.contains FastRows.row4861) := by
    change Data.row4861.contains = (SortedNatTree.contains FastRows.row4861)
    exact FastRows.row4861_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child41 : search edge 3 [19, 4861] root5Candidates41 4880 winner = winner := by decide
theorem root5Step41 : scan edge (search edge 3) 4 [19] 19 root5Tail41 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail42 winner := by
  rw [root5Tail41, scan, if_neg (by decide), root5Candidates41_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail42
    (if 3 ≤ root5Candidates41.length then search edge 3 [19, 4861] root5Candidates41 4880 winner else winner) = _
  rw [if_pos (by decide), root5Child41]
def root5Candidates42 : List Nat := [6043, 10663]
set_option maxRecDepth 8192 in
theorem root5Candidates42_checked : candidates edge 4 19 5077 root5Tail43 winner.bound = root5Candidates42 := by
  unfold candidates
  have row : edge 5077 = (SortedNatTree.contains FastRows.row5077) := by
    change Data.row5077.contains = (SortedNatTree.contains FastRows.row5077)
    exact FastRows.row5077_checked.symm
  rw [row]
  decide
theorem root5Step42 : scan edge (search edge 3) 4 [19] 19 root5Tail42 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail43 winner := by
  rw [root5Tail42, scan, if_neg (by decide), root5Candidates42_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail43
    (if 3 ≤ root5Candidates42.length then search edge 3 [19, 5077] root5Candidates42 5096 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates43 : List Nat := [6271, 7759, 10567]
set_option maxRecDepth 8192 in
theorem root5Candidates43_checked : candidates edge 4 19 5659 root5Tail44 winner.bound = root5Candidates43 := by
  unfold candidates
  have row : edge 5659 = (SortedNatTree.contains FastRows.row5659) := by
    change Data.row5659.contains = (SortedNatTree.contains FastRows.row5659)
    exact FastRows.row5659_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child43 : search edge 3 [19, 5659] root5Candidates43 5678 winner = winner := by decide
theorem root5Step43 : scan edge (search edge 3) 4 [19] 19 root5Tail43 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail44 winner := by
  rw [root5Tail43, scan, if_neg (by decide), root5Candidates43_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail44
    (if 3 ≤ root5Candidates43.length then search edge 3 [19, 5659] root5Candidates43 5678 winner else winner) = _
  rw [if_pos (by decide), root5Child43]
def root5Candidates44 : List Nat := [8929, 8941, 14389, 14593]
set_option maxRecDepth 8192 in
theorem root5Candidates44_checked : candidates edge 4 19 5737 root5Tail45 winner.bound = root5Candidates44 := by
  unfold candidates
  have row : edge 5737 = (SortedNatTree.contains FastRows.row5737) := by
    change Data.row5737.contains = (SortedNatTree.contains FastRows.row5737)
    exact FastRows.row5737_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child44 : search edge 3 [19, 5737] root5Candidates44 5756 winner = winner := by decide
theorem root5Step44 : scan edge (search edge 3) 4 [19] 19 root5Tail44 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail45 winner := by
  rw [root5Tail44, scan, if_neg (by decide), root5Candidates44_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail45
    (if 3 ≤ root5Candidates44.length then search edge 3 [19, 5737] root5Candidates44 5756 winner else winner) = _
  rw [if_pos (by decide), root5Child44]
def root5Candidates45 : List Nat := [9601, 17569, 18517]
set_option maxRecDepth 8192 in
theorem root5Candidates45_checked : candidates edge 4 19 5791 root5Tail46 winner.bound = root5Candidates45 := by
  unfold candidates
  have row : edge 5791 = (SortedNatTree.contains FastRows.row5791) := by
    change Data.row5791.contains = (SortedNatTree.contains FastRows.row5791)
    exact FastRows.row5791_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child45 : search edge 3 [19, 5791] root5Candidates45 5810 winner = winner := by decide
theorem root5Step45 : scan edge (search edge 3) 4 [19] 19 root5Tail45 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail46 winner := by
  rw [root5Tail45, scan, if_neg (by decide), root5Candidates45_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail46
    (if 3 ≤ root5Candidates45.length then search edge 3 [19, 5791] root5Candidates45 5810 winner else winner) = _
  rw [if_pos (by decide), root5Child45]
def root5Candidates46 : List Nat := [6709, 6961, 12577]
set_option maxRecDepth 8192 in
theorem root5Candidates46_checked : candidates edge 4 19 5869 root5Tail47 winner.bound = root5Candidates46 := by
  unfold candidates
  have row : edge 5869 = (SortedNatTree.contains FastRows.row5869) := by
    change Data.row5869.contains = (SortedNatTree.contains FastRows.row5869)
    exact FastRows.row5869_checked.symm
  rw [row]
  decide
set_option maxRecDepth 4096 in
theorem root5Child46 : search edge 3 [19, 5869] root5Candidates46 5888 winner = winner := by decide
theorem root5Step46 : scan edge (search edge 3) 4 [19] 19 root5Tail46 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail47 winner := by
  rw [root5Tail46, scan, if_neg (by decide), root5Candidates46_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail47
    (if 3 ≤ root5Candidates46.length then search edge 3 [19, 5869] root5Candidates46 5888 winner else winner) = _
  rw [if_pos (by decide), root5Child46]
def root5Candidates47 : List Nat := [16183]
set_option maxRecDepth 8192 in
theorem root5Candidates47_checked : candidates edge 4 19 6043 root5Tail48 winner.bound = root5Candidates47 := by
  unfold candidates
  have row : edge 6043 = (SortedNatTree.contains FastRows.row6043) := by
    change Data.row6043.contains = (SortedNatTree.contains FastRows.row6043)
    exact FastRows.row6043_checked.symm
  rw [row]
  decide
theorem root5Step47 : scan edge (search edge 3) 4 [19] 19 root5Tail47 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail48 winner := by
  rw [root5Tail47, scan, if_neg (by decide), root5Candidates47_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail48
    (if 3 ≤ root5Candidates47.length then search edge 3 [19, 6043] root5Candidates47 6062 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates48 : List Nat := [7759, 19387]
set_option maxRecDepth 8192 in
theorem root5Candidates48_checked : candidates edge 4 19 6073 root5Tail49 winner.bound = root5Candidates48 := by
  unfold candidates
  have row : edge 6073 = (SortedNatTree.contains FastRows.row6073) := by
    change Data.row6073.contains = (SortedNatTree.contains FastRows.row6073)
    exact FastRows.row6073_checked.symm
  rw [row]
  decide
theorem root5Step48 : scan edge (search edge 3) 4 [19] 19 root5Tail48 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail49 winner := by
  rw [root5Tail48, scan, if_neg (by decide), root5Candidates48_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail49
    (if 3 ≤ root5Candidates48.length then search edge 3 [19, 6073] root5Candidates48 6092 winner else winner) = _
  rw [if_neg (by decide)]
def root5Candidates49 : List Nat := [8641]
set_option maxRecDepth 8192 in
theorem root5Candidates49_checked : candidates edge 4 19 6271 root5Tail50 winner.bound = root5Candidates49 := by
  unfold candidates
  have row : edge 6271 = (SortedNatTree.contains FastRows.row6271) := by
    change Data.row6271.contains = (SortedNatTree.contains FastRows.row6271)
    exact FastRows.row6271_checked.symm
  rw [row]
  decide
theorem root5Step49 : scan edge (search edge 3) 4 [19] 19 root5Tail49 winner =
    scan edge (search edge 3) 4 [19] 19 root5Tail50 winner := by
  rw [root5Tail49, scan, if_neg (by decide), root5Candidates49_checked]
  change scan edge (search edge 3) 4 [19] 19 root5Tail50
    (if 3 ≤ root5Candidates49.length then search edge 3 [19, 6271] root5Candidates49 6290 winner else winner) = _
  rw [if_neg (by decide)]
end ProjectEulerSolutions.PrimePairLarge.Search
