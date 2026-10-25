import ProjectEulerSolutions.Lib.CubicWindows

/-! Checkable partitions of cube roots supply exact group counts without replaying the search. -/
namespace ProjectEulerSolutions.CubicPartition
open CubicPermutations CubicGroups CubicGroupCounts CubicSearch CubicWindows
set_option Elab.async false

/-- The index is only a search aid; checking the rows establishes their actual signatures. -/
def code (digits : List Nat) : Nat := digits.foldr (fun digit rest => digit + 14 * rest) 0

structure Row where
  key : Nat
  digits : List Nat
  roots : List Nat
deriving DecidableEq

def Row.group (row : Row) : Group := ⟨row.digits, row.roots.length, cube (row.roots.headD 0)⟩

inductive Tree where
  | empty
  | node (row : Row) (left right : Tree)

def entries : Tree → List Row
  | .empty => []
  | .node row left right => entries left ++ row :: entries right

def lookup : Tree → Nat → Option Row
  | .empty, _ => none
  | .node row left right, key =>
      if key < row.key then lookup left key else if row.key < key then lookup right key else some row

def ordered : Tree → Bool
  | .empty => true
  | .node row left right =>
      (ordered left && ordered right) &&
        ((entries left).all (fun other => decide (other.key < row.key)) &&
          (entries right).all (fun other => decide (row.key < other.key)))

theorem ordered_node (row : Row) (left right : Tree) (before : ordered left = true) (after : ordered right = true)
    (bounds : ((entries left).all (fun other => decide (other.key < row.key)) &&
      (entries right).all (fun other => decide (row.key < other.key))) = true) :
    ordered (.node row left right) = true := by
  simp only [ordered, before, after, bounds, Bool.true_and]

theorem lookup_mem (tree : Tree) (key : Nat) {row : Row} (found : lookup tree key = some row) :
    row ∈ entries tree := by
  induction tree with
  | empty => simp [lookup] at found
  | node middle left right ihLeft ihRight =>
      simp only [lookup] at found
      split at found
      · exact List.mem_append_left _ (ihLeft found)
      · split at found
        · exact List.mem_append_right _ (List.mem_cons_of_mem _ (ihRight found))
        · have equal := Option.some.inj found
          simp [entries, equal]

theorem lookup_entry (tree : Tree) (valid : ordered tree = true) (row : Row) (member : row ∈ entries tree) :
    lookup tree row.key = some row := by
  induction tree with
  | empty => simp [entries] at member
  | node middle left right ihLeft ihRight =>
      simp only [ordered, Bool.and_eq_true] at valid
      have before : ∀ other ∈ entries left, other.key < middle.key := by
        simpa only [List.all_eq_true, decide_eq_true_eq] using valid.2.1
      have after : ∀ other ∈ entries right, middle.key < other.key := by
        simpa only [List.all_eq_true, decide_eq_true_eq] using valid.2.2
      rcases List.mem_append.mp member with earlier | here
      · simp only [lookup, if_pos (before row earlier)]
        exact ihLeft valid.1.1 earlier
      · rcases List.mem_cons.mp here with same | later
        · subst row
          simp [lookup]
        · have greater := after row later
          simp only [lookup, if_neg (by omega : ¬ row.key < middle.key), if_pos greater]
          exact ihRight valid.1.2 later

theorem entries_ordered (tree : Tree) (valid : ordered tree = true) :
    (entries tree).Pairwise (fun left right => left.key < right.key) := by
  induction tree with
  | empty => simp [entries]
  | node row left right ihLeft ihRight =>
      simp only [ordered, Bool.and_eq_true] at valid
      have before : ∀ other ∈ entries left, other.key < row.key := by
        simpa only [List.all_eq_true, decide_eq_true_eq] using valid.2.1
      have after : ∀ other ∈ entries right, row.key < other.key := by
        simpa only [List.all_eq_true, decide_eq_true_eq] using valid.2.2
      simp only [entries, List.pairwise_append, List.pairwise_cons]
      refine ⟨ihLeft valid.1.1, ⟨after, ihRight valid.1.2⟩, ?_⟩
      intro earlier earlierMember later laterMember
      rcases List.mem_cons.mp laterMember with same | laterMember
      · subst later
        exact before earlier earlierMember
      · exact (before earlier earlierMember).trans (after later laterMember)

structure RowValid (bound : Nat) (row : Row) : Prop where
  key : row.key = code row.digits
  nonempty : row.roots ≠ []
  sorted : row.roots.Pairwise (· < ·)
  members : ∀ root ∈ row.roots, 0 < root ∧ root ≤ bound ∧ signature (cube root) = row.digits

def rowChecked (bound : Nat) (row : Row) : Bool :=
  decide (row.key = code row.digits) && decide (row.roots ≠ []) && decide (row.roots.Pairwise (· < ·)) &&
    row.roots.all (fun root => decide (0 < root ∧ root ≤ bound ∧ signature (cube root) = row.digits))

theorem rowChecked_sound (bound : Nat) (row : Row) (checked : rowChecked bound row = true) : RowValid bound row := by
  simp only [rowChecked, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at checked
  exact ⟨checked.1.1.1, checked.1.1.2, checked.1.2, checked.2⟩

def rowsChecked (bound : Nat) : Tree → Bool
  | .empty => true
  | .node row left right => rowChecked bound row && (rowsChecked bound left && rowsChecked bound right)

theorem rowsChecked_node (bound : Nat) (row : Row) (left right : Tree) (here : rowChecked bound row = true)
    (before : rowsChecked bound left = true) (after : rowsChecked bound right = true) :
    rowsChecked bound (.node row left right) = true := by
  simp only [rowsChecked, here, before, after, Bool.true_and]

theorem rowsChecked_sound (bound : Nat) (tree : Tree) (checked : rowsChecked bound tree = true) :
    ∀ row ∈ entries tree, RowValid bound row := by
  induction tree with
  | empty => simp [entries]
  | node middle left right ihLeft ihRight =>
      simp only [rowsChecked, Bool.and_eq_true] at checked
      intro row member
      rcases List.mem_append.mp member with earlier | here
      · exact ihLeft checked.2.1 row earlier
      · rcases List.mem_cons.mp here with same | later
        · subst row
          exact rowChecked_sound bound middle checked.1
        · exact ihRight checked.2.2 row later

def covered (tree : Tree) (root : Nat) : Bool :=
  match lookup tree (code (signature (cube root))) with
  | none => false
  | some row => row.roots.contains root

def Covers (tree : Tree) (first count : Nat) : Prop :=
  ∀ root, first ≤ root → root < first + count → covered tree root = true

theorem covers_checked (tree : Tree) (first count : Nat)
    (checked : (List.range' first count).all (covered tree) = true) : Covers tree first count := by
  intro root lower upper
  exact List.all_eq_true.mp checked root (List.mem_range'_1.mpr ⟨lower, upper⟩)

theorem Covers.append {tree : Tree} {first left right : Nat}
    (before : Covers tree first left) (after : Covers tree (first + left) right) :
    Covers tree first (left + right) := by
  intro root lower upper
  by_cases early : root < first + left
  · exact before root lower early
  · exact after root (by omega) (by omega)

structure Valid (bound : Nat) (tree : Tree) : Prop where
  ordered : ordered tree = true
  rows : ∀ row ∈ entries tree, RowValid bound row
  cover : Covers tree 1 bound

theorem Valid.root_class {bound : Nat} {tree : Tree} (valid : Valid bound tree) (root : Nat)
    (positive : 0 < root) (bounded : root ≤ bound) :
    ∃ row ∈ entries tree, root ∈ row.roots ∧ row.digits = signature (cube root) := by
  have coveredRoot := valid.cover root (by omega) (by omega)
  unfold covered at coveredRoot
  cases found : lookup tree (code (signature (cube root))) with
  | none => simp [found] at coveredRoot
  | some row =>
      have member := lookup_mem tree _ found
      have rootMember : root ∈ row.roots := by simpa [found] using coveredRoot
      exact ⟨row, member, rootMember, ((valid.rows row member).members root rootMember).2.2.symm⟩

theorem Valid.exact_class {bound : Nat} {tree : Tree} (valid : Valid bound tree) (row : Row)
    (member : row ∈ entries tree) (root : Nat) :
    root ∈ row.roots ↔ 0 < root ∧ root ≤ bound ∧ signature (cube root) = row.digits := by
  constructor
  · exact (valid.rows row member).members root
  · rintro ⟨positive, bounded, same⟩
    have checked := valid.cover root (by omega) (by omega)
    have found : lookup tree (code (signature (cube root))) = some row := by
      rw [same, ← (valid.rows row member).key]
      exact lookup_entry tree valid.ordered row member
    simpa only [covered, found, List.contains_iff_mem] using checked

theorem Valid.root_counts {bound : Nat} {tree : Tree} (valid : Valid bound tree) (row : Row)
    (member : row ∈ entries tree) :
    row.roots.length = occurrences row.digits (prior (bound + 1)) := by
  have rootNodup := (valid.rows row member).sorted.nodup
  have rangeNodup : (List.range' 1 bound).Nodup := (List.sortedLT_range' 1 bound (by decide)).pairwise.nodup
  have sameMembers : ∀ root, root ∈ row.roots ↔
      root ∈ (List.range' 1 bound).filter (fun root => decide (signature (cube root) = row.digits)) := by
    intro root
    rw [valid.exact_class row member root]
    simp only [List.mem_filter, List.mem_range'_1, decide_eq_true_eq]
    constructor
    · rintro ⟨positive, upper, same⟩
      exact ⟨⟨by omega, by omega⟩, same⟩
    · rintro ⟨⟨lower, upper⟩, same⟩
      exact ⟨by omega, by omega, same⟩
  have sameCounts : ∀ root, row.roots.count root =
      ((List.range' 1 bound).filter (fun root => decide (signature (cube root) = row.digits))).count root := by
    intro root
    by_cases present : root ∈ row.roots
    · rw [List.count_eq_one_of_mem rootNodup present,
        List.count_eq_one_of_mem (rangeNodup.filter _) ((sameMembers root).mp present)]
    · rw [List.count_eq_zero.mpr present, List.count_eq_zero.mpr (fun other => present ((sameMembers root).mpr other))]
  have permutation := List.perm_iff_count.mpr sameCounts
  simpa only [occurrences, prior, Nat.add_sub_cancel, List.filter_map, List.length_map, Function.comp_def] using
    permutation.length_eq

/-- Coverage and checked rows establish a complete count/minimum summary of the finite window. -/
theorem Valid.summary {bound : Nat} {tree : Tree} (valid : Valid bound tree) :
    Summary (prior (bound + 1)) ((entries tree).map Row.group) := by
  constructor
  · rw [List.map_map]
    have pairwise := entries_ordered tree valid.ordered
    change ((entries tree).map (fun row => row.digits)).Pairwise (· ≠ ·)
    rw [List.pairwise_map]
    apply pairwise.imp_of_mem
    intro left right leftMember rightMember smaller same
    have leftKey := (valid.rows left leftMember).key
    have rightKey := (valid.rows right rightMember).key
    change left.digits = right.digits at same
    rw [leftKey, rightKey, same] at smaller
    exact Nat.lt_irrefl _ smaller
  · intro value member
    obtain ⟨root, positive, upper, rfl⟩ := mem_prior.mp member
    obtain ⟨row, rowMember, _, same⟩ := valid.root_class root positive (by omega)
    exact ⟨row.group, List.mem_map.mpr ⟨row, rowMember, rfl⟩, same⟩
  · intro group member
    obtain ⟨row, rowMember, rfl⟩ := List.mem_map.mp member
    change (row.roots.length : Int) = (occurrences row.digits (prior (bound + 1)) : Int)
    exact_mod_cast valid.root_counts row rowMember
  · intro group member
    obtain ⟨row, rowMember, rfl⟩ := List.mem_map.mp member
    have rowValid := valid.rows row rowMember
    cases rootsEq : row.roots with
    | nil => exact False.elim (rowValid.nonempty rootsEq)
    | cons first rest =>
        have firstMember : first ∈ row.roots := by rw [rootsEq]; simp
        obtain ⟨positive, bounded, same⟩ := rowValid.members first firstMember
        refine ⟨cube first, mem_prior.mpr ⟨first, positive, by omega, rfl⟩, same, ?_, ?_⟩
        · simp only [Row.group, rootsEq, List.headD_cons]
        · intro other otherMember otherKey
          obtain ⟨root, rootPositive, upper, rfl⟩ := mem_prior.mp otherMember
          have rootMember := (valid.exact_class row rowMember root).mpr ⟨rootPositive, by omega, otherKey⟩
          have ordered : first ≤ root := by
            rw [rootsEq] at rootMember
            rcases List.mem_cons.mp rootMember with equal | later
            · omega
            · have sorted := rowValid.sorted
              rw [rootsEq, List.pairwise_cons] at sorted
              exact (sorted.1 root later).le
          simpa only [Row.group, rootsEq, List.headD_cons] using
            (show (cube first : Int) ≤ (cube root : Int) by exact_mod_cast cube_mono ordered)

def candidates (target : Int) : Tree → List Int
  | .empty => []
  | .node row left right => candidates target left ++
      (if (row.roots.length : Int) = target then [(cube (row.roots.headD 0) : Int)] else []) ++ candidates target right

theorem matching_cons (target : Int) (group : Group) (groups : List Group) :
    matching target (group :: groups) = (if group.count = target then [group.minimum] else []) ++ matching target groups := by
  by_cases selected : group.count = target <;> simp [matching, selected]

theorem matching_candidates (target : Int) (tree : Tree) :
    matching target ((entries tree).map Row.group) = candidates target tree := by
  induction tree with
  | empty => rfl
  | node row left right ihLeft ihRight =>
      simp only [entries, List.map_append, List.map_cons, CubicWindows.matching_append, matching_cons,
        ihLeft, ihRight, candidates, Row.group, List.append_assoc]

theorem candidates_node (target : Int) (row : Row) (left right : Tree) (before after : List Int)
    (leftChecked : candidates target left = before) (rightChecked : candidates target right = after) :
    candidates target (.node row left right) = before ++
      (if (row.roots.length : Int) = target then [(cube (row.roots.headD 0) : Int)] else []) ++ after := by
  simp only [candidates, leftChecked, rightChecked]

end ProjectEulerSolutions.CubicPartition
