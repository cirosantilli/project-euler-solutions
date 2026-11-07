import Mathlib.Data.List.FinRange
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Prod

/-! Complete finite Sudoku search, independent of the order of empty cells. -/
namespace ProjectEulerSolutions.Sudoku

set_option Elab.async false

abbrev Cell := Fin 9 × Fin 9
abbrev Digit := Fin 9
abbrev Board := Cell → Option Digit

def Peer (a b : Cell) : Prop :=
  a ≠ b ∧ (a.1 = b.1 ∨ a.2 = b.2 ∨
    (a.1.val / 3 = b.1.val / 3 ∧ a.2.val / 3 = b.2.val / 3))

instance (a b : Cell) : Decidable (Peer a b) := inferInstanceAs (Decidable (_ ∧ _))

theorem Peer.symm {a b : Cell} (peer : Peer a b) : Peer b a := by
  refine ⟨Ne.symm peer.1, ?_⟩
  rcases peer.2 with row | column | box
  · exact Or.inl row.symm
  · exact Or.inr (Or.inl column.symm)
  · exact Or.inr (Or.inr ⟨box.1.symm, box.2.symm⟩)

def Valid (board : Board) : Prop :=
  ∀ a b, Peer a b → ∀ digit, board a = some digit → board b ≠ some digit

def Full (board : Board) : Prop := ∀ cell, board cell ≠ none

def Extends (initial final : Board) : Prop :=
  ∀ cell digit, initial cell = some digit → final cell = some digit

def Completion (initial final : Board) : Prop :=
  Valid final ∧ Full final ∧ Extends initial final

def Allowed (board : Board) (cell : Cell) (digit : Digit) : Prop :=
  ∀ other, Peer cell other → board other ≠ some digit

instance (board : Board) (cell : Cell) (digit : Digit) : Decidable (Allowed board cell digit) :=
  inferInstanceAs (Decidable (∀ other, Peer cell other → board other ≠ some digit))

theorem valid_update {board : Board} {cell : Cell} {digit : Digit}
    (valid : Valid board) (allowed : Allowed board cell digit) :
    Valid (Function.update board cell (some digit)) := by
  intro a b peer value atA atB
  by_cases left : a = cell
  · subst a
    simp only [Function.update_self, Option.some.injEq] at atA
    subst value
    rw [Function.update_of_ne (Ne.symm peer.1)] at atB
    exact allowed b peer atB
  · rw [Function.update_of_ne left] at atA
    by_cases right : b = cell
    · subst b
      simp only [Function.update_self, Option.some.injEq] at atB
      subst value
      exact allowed a peer.symm atA
    · rw [Function.update_of_ne right] at atB
      exact valid a b peer value atA atB

theorem extends_trans {a b c : Board} (first : Extends a b) (second : Extends b c) :
    Extends a c := fun cell digit found => second cell digit (first cell digit found)

theorem extends_update {board : Board} {cell : Cell} (empty : board cell = none) (digit : Digit) :
    Extends board (Function.update board cell (some digit)) := by
  intro other value found
  have distinct : other ≠ cell := by
    intro same
    subst other
    simp [empty] at found
  simpa [Function.update_of_ne distinct] using found

theorem completion_allowed {board final : Board} {cell : Cell} {digit : Digit}
    (completed : Completion board final) (found : final cell = some digit) :
    Allowed board cell digit := by
  intro other peer original
  exact completed.1 cell other peer digit found (completed.2.2 other digit original)

theorem completion_update {board final : Board} {cell : Cell} {digit : Digit}
    (completed : Completion board final) (found : final cell = some digit) :
    Completion (Function.update board cell (some digit)) final := by
  refine ⟨completed.1, completed.2.1, ?_⟩
  intro other value original
  by_cases same : other = cell
  · subst other
    simp only [Function.update_self, Option.some.injEq] at original
    simpa [← original] using found
  · rw [Function.update_of_ne same] at original
    exact completed.2.2 other value original

theorem full_extends_eq {initial final : Board} (full : Full initial)
    (extension : Extends initial final) : final = initial := by
  funext cell
  cases found : initial cell with
  | none => exact False.elim (full cell found)
  | some digit => exact extension cell digit found

/-- Nine different filled cells in a unit contain every one of the nine digits. -/
theorem unit_contains_every_digit {board : Board} (valid : Valid board) (full : Full board)
    (unit : Fin 9 → Cell) (peers : ∀ i j, i ≠ j → Peer (unit i) (unit j))
    (digit : Digit) : ∃ index, board (unit index) = some digit := by
  have found (cell : Cell) : board cell = some ((board cell).getD 0) := by
    cases value : board cell with
    | none => exact False.elim (full cell value)
    | some value => rfl
  have injective : Function.Injective (fun i => (board (unit i)).getD 0) := by
    intro i j same
    by_contra distinct
    exact valid (unit i) (unit j) (peers i j distinct) _ (found (unit i))
      (by simpa [same] using found (unit j))
  obtain ⟨index, equal⟩ := Finite.surjective_of_injective injective digit
  exact ⟨index, by simpa [equal] using found (unit index)⟩

theorem row_contains_every_digit {board : Board} (valid : Valid board) (full : Full board)
    (row : Fin 9) (digit : Digit) : ∃ column, board (row, column) = some digit := by
  apply unit_contains_every_digit valid full (fun column => (row, column)) _ digit
  intro i j distinct
  exact ⟨by simpa using distinct, Or.inl rfl⟩

theorem column_contains_every_digit {board : Board} (valid : Valid board) (full : Full board)
    (column : Fin 9) (digit : Digit) : ∃ row, board (row, column) = some digit := by
  apply unit_contains_every_digit valid full (fun row => (row, column)) _ digit
  intro i j distinct
  exact ⟨by simpa using distinct, Or.inr (Or.inl rfl)⟩

def boxCell (row column : Fin 3) (index : Fin 9) : Cell :=
  (⟨3 * row.val + index.val / 3, by omega⟩,
   ⟨3 * column.val + index.val % 3, by omega⟩)

theorem boxCell_peer (row column : Fin 3) (i j : Fin 9) (distinct : i ≠ j) :
    Peer (boxCell row column i) (boxCell row column j) := by
  constructor
  · intro same
    have equalRow := congrArg (fun cell : Cell => cell.1.val) same
    have equalColumn := congrArg (fun cell : Cell => cell.2.val) same
    simp only [boxCell] at equalRow equalColumn
    apply distinct
    apply Fin.ext
    omega
  · right
    right
    simp only [boxCell]
    constructor <;> omega

theorem box_contains_every_digit {board : Board} (valid : Valid board) (full : Full board)
    (row column : Fin 3) (digit : Digit) : ∃ index, board (boxCell row column index) = some digit :=
  unit_contains_every_digit valid full (boxCell row column) (boxCell_peer row column) digit

/-- Exhaustive search visits every digit at each empty cell, in any supplied cell order. -/
def completions : List Cell → Board → List Board
  | [], board => [board]
  | cell :: rest, board =>
      if board cell = none then
        (List.finRange 9).flatMap fun digit =>
          if Allowed board cell digit then completions rest (Function.update board cell (some digit))
          else []
      else completions rest board

theorem completions_spec (pending : List Cell) (board : Board)
    (valid : Valid board) (covered : ∀ cell, board cell = none → cell ∈ pending) (final : Board) :
    final ∈ completions pending board ↔ Completion board final := by
  induction pending generalizing board with
  | nil =>
      have full : Full board := by
        intro cell empty
        simpa using covered cell empty
      simp only [completions, List.mem_singleton]
      constructor
      · intro same
        subst final
        exact ⟨valid, full, fun _ _ found => found⟩
      · intro completed
        exact full_extends_eq full completed.2.2
  | cons cell rest ih =>
      by_cases empty : board cell = none
      · rw [completions, if_pos empty]
        constructor
        · intro member
          obtain ⟨digit, _, member⟩ := List.mem_flatMap.mp member
          split at member
          next allowed =>
            have covered' : ∀ other, Function.update board cell (some digit) other = none → other ∈ rest := by
              intro other missing
              have distinct : other ≠ cell := by intro same; subst other; simp at missing
              rw [Function.update_of_ne distinct] at missing
              exact (List.mem_cons.mp (covered other missing)).resolve_left distinct
            have completed := (ih _ (valid_update valid allowed) covered').mp member
            exact ⟨completed.1, completed.2.1, extends_trans (extends_update empty digit) completed.2.2⟩
          next => simp at member
        · intro completed
          cases found : final cell with
          | none => exact False.elim (completed.2.1 cell found)
          | some digit =>
              apply List.mem_flatMap.mpr
              refine ⟨digit, List.mem_finRange _, ?_⟩
              have allowed := completion_allowed completed found
              rw [if_pos allowed]
              have covered' : ∀ other, Function.update board cell (some digit) other = none → other ∈ rest := by
                intro other missing
                have distinct : other ≠ cell := by intro same; subst other; simp at missing
                rw [Function.update_of_ne distinct] at missing
                exact (List.mem_cons.mp (covered other missing)).resolve_left distinct
              exact (ih _ (valid_update valid allowed) covered').mpr (completion_update completed found)
      · rw [completions, if_neg empty]
        apply ih board valid
        intro other missing
        have distinct : other ≠ cell := by intro same; subst other; exact empty missing
        exact (List.mem_cons.mp (covered other missing)).resolve_left distinct

/-- Uniqueness makes every successful search order return the same solution. -/
theorem unique_result {board expected : Board}
    (unique : ∀ final, Completion board final → final = expected)
    (pending : List Cell) (valid : Valid board)
    (covered : ∀ cell, board cell = none → cell ∈ pending)
    {actual : Board} (returned : actual ∈ completions pending board) : actual = expected :=
  unique actual ((completions_spec pending board valid covered actual).mp returned)

/-- Stop at the first successful branch, as in the reference backtracking loop. -/
def search : List Cell → Board → Option Board
  | [], board => some board
  | cell :: rest, board =>
      if board cell = none then
        (List.finRange 9).findSome? fun digit =>
          if Allowed board cell digit then search rest (Function.update board cell (some digit))
          else none
      else search rest board

theorem search_eq_head (pending : List Cell) (board : Board) :
    search pending board = (completions pending board).head? := by
  induction pending generalizing board with
  | nil => rfl
  | cons cell rest ih =>
      simp only [search, completions]
      split
      · rw [List.head?_flatMap]
        congr 1
        funext digit
        split <;> simp [ih]
      · exact ih board

theorem search_sound {pending : List Cell} {board final : Board}
    (valid : Valid board) (covered : ∀ cell, board cell = none → cell ∈ pending)
    (returned : search pending board = some final) : Completion board final := by
  rw [search_eq_head] at returned
  exact (completions_spec pending board valid covered final).mp (List.mem_of_mem_head? returned)

theorem search_none_iff (pending : List Cell) (board : Board)
    (valid : Valid board) (covered : ∀ cell, board cell = none → cell ∈ pending) :
    search pending board = none ↔ ¬ ∃ final, Completion board final := by
  rw [search_eq_head, List.head?_eq_none_iff]
  constructor
  · intro empty ⟨final, completed⟩
    have member := (completions_spec pending board valid covered final).mpr completed
    simp [empty] at member
  · intro impossible
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro final member
    exact impossible ⟨final, (completions_spec pending board valid covered final).mp member⟩

theorem search_unique {board expected : Board} (completed : Completion board expected)
    (unique : ∀ final, Completion board final → final = expected)
    (pending : List Cell) (valid : Valid board)
    (covered : ∀ cell, board cell = none → cell ∈ pending) :
    search pending board = some expected := by
  cases returned : search pending board with
  | none => exact False.elim ((search_none_iff pending board valid covered).mp returned ⟨expected, completed⟩)
  | some actual =>
      have same := unique actual (search_sound valid covered returned)
      exact congrArg some same

end ProjectEulerSolutions.Sudoku
