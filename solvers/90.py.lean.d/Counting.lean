import solvers.«90.py.lean.d».Display
import solvers.«90.py.lean.d».Masks

/-! The nested index loops count unordered pairs, including identical printed designs. -/
namespace solvers.«90».counting
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions DigitDice
set_option Elab.async false

structure Data where
  masks : List Nat
  pairs : List can_display.Pair
  masksAddress : Nat
  pairsAddress : Nat

def Data.test (data : Data) (left right : Nat) : Bool :=
  maskDisplays left right (data.pairs.map can_display.Pair.model)

structure Valid (data : Data) (heap : Mutable.Heap) : Prop where
  masks : heap[data.masksAddress]? = some (.list (data.masks.map number))
  pairs : heap[data.pairsAddress]? = some (.list (data.pairs.map can_display.Pair.reference))
  tuples : ∀ pair ∈ data.pairs, pair.Valid heap

structure Ready (data : Data) (env : HeapCalls.Env) : Prop where
  masks : env "expanded" = some (.ref data.masksAddress)
  pairs : env "required_pairs" = some (.ref data.pairsAddress)
  size : env "n" = some (number data.masks.length)

theorem Ready.set_auxiliary {data : Data} {env : HeapCalls.Env} (ready : Ready data env)
    (name : String) (value : Mutable.Value) (notMasks : name ≠ "expanded")
    (notPairs : name ≠ "required_pairs") (notSize : name ≠ "n") :
    Ready data (setLocal env name value) := by
  constructor
  · simpa [setLocal, Ne.symm notMasks] using ready.masks
  · simpa [setLocal, Ne.symm notPairs] using ready.pairs
  · simpa [setLocal, Ne.symm notSize] using ready.size

def outerCode : List Stmt :=
  match (masks.code[6]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

def innerCode : List Stmt :=
  match (outerCode[0]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private theorem index_evaluates (data : Data) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (valid : Valid data heap) (ready : Ready data env)
    (name : String) (index value : Nat) (getIndex : env name = some (number index))
    (item : data.masks[index]? = some value) :
    Evaluates program env (.subscript (.name "expanded") (.name name)) heap (number value) heap globals := by
  obtain ⟨inside, same⟩ := List.getElem?_eq_some_iff.mp item
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup ready.masks) (Evaluates.lookup getIndex)
  have read := Mutable.getItem_list heap data.masksAddress index (data.masks.map number) valid.masks
    (by simpa using inside)
  simpa only [List.getElem_map, same] using read

private theorem test_evaluates (data : Data) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (valid : Valid data heap) (ready : Ready data env)
    (i j left right : Nat) (getI : env "i" = some (number i)) (getJ : env "j" = some (number j))
    (leftItem : data.masks[i]? = some left) (rightItem : data.masks[j]? = some right) :
    Evaluates program env (.call (.name "can_display") [
      .subscript (.name "expanded") (.name "i"), .subscript (.name "expanded") (.name "j"),
      .name "required_pairs"]) heap (.scalar (.bool (data.test left right))) heap globals :=
  Evaluates.call (.cons (index_evaluates data heap env globals valid ready "i" i left getI leftItem)
    (.cons (index_evaluates data heap env globals valid ready "j" j right getJ rightItem)
      (.cons (Evaluates.lookup ready.pairs) .nil)))
    (can_display.result left right data.pairsAddress data.pairs heap valid.pairs valid.tuples globals)

/-- A row counts exactly the accepted partners from its current index through the end. -/
theorem row_loop (data : Data) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (valid : Valid data heap) (i left : Nat) (leftItem : data.masks[i]? = some left)
    (remaining : List Nat) (position count : Nat) (env : HeapCalls.Env)
    (suffix : data.masks.drop position = remaining) (ready : Ready data env)
    (getI : env "i" = some (number i)) (getCount : env "count" = some (number count)) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "j" position data.masks.length 1 innerCode []] heap finalEnv heap globals ∧
      Ready data finalEnv ∧ finalEnv "count" = some (number (count + remaining.countP (data.test left))) := by
  induction remaining generalizing position count env with
  | nil =>
      refine ⟨env, ?_, ready, ?_⟩
      · apply BlockContinues.range_done
        · have finished := List.drop_eq_nil_iff.mp suffix
          simp [Calls.rangeContinues]
          omega
        · exact BlockContinues.nil
      · simpa using getCount
  | cons right rest ih =>
      have item : data.masks[position]? = some right := by
        simpa [suffix] using (List.head?_drop (l := data.masks) (i := position)).symm
      have tail : data.masks.drop (position + 1) = rest := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      let bound := setLocal env "j" (number position)
      have boundReady : Ready data bound := ready.set_auxiliary "j" _ (by decide) (by decide) (by decide)
      have boundI : bound "i" = some (number i) := by simpa [bound, setLocal] using getI
      have boundCount : bound "count" = some (number count) := by simpa [bound, setLocal] using getCount
      have tested := test_evaluates data heap bound globals valid boundReady i position left right boundI rfl leftItem item
      let updated := if data.test left right then setLocal bound "count" (number (count + 1)) else bound
      have updatedReady : Ready data updated := by
        cases accepted : data.test left right with
        | false => simpa [updated, accepted] using boundReady
        | true => simpa [updated, accepted] using
            boundReady.set_auxiliary "count" (number (count + 1)) (by decide) (by decide) (by decide)
      have updatedI : updated "i" = some (number i) := by
        cases accepted : data.test left right <;> simpa [updated, accepted, setLocal] using boundI
      have updatedCount : updated "count" = some (number (count + (data.test left right).toNat)) := by
        cases accepted : data.test left right <;> simp [updated, accepted, setLocal, boundCount]
      have visit : BlockContinues program bound innerCode heap updated heap globals := by
        cases accepted : data.test left right with
        | false =>
            simp only [updated, accepted, Bool.false_eq_true, ↓reduceIte]
            apply BlockContinues.branch (flag := false) (by simpa only [accepted] using tested) rfl
            exact BlockContinues.nil
        | true =>
            simp only [updated, accepted, ↓reduceIte]
            apply BlockContinues.branch (flag := true) (by simpa only [accepted] using tested) rfl
            apply BlockContinues.augAssign boundCount Evaluates.constant
            simp [inplace, Mutable.binary, scalarBinary, number]
      obtain ⟨finalEnv, continued, finalReady, result⟩ :=
        ih (position + 1) (count + (data.test left right).toNat) updated tail updatedReady updatedI updatedCount
      refine ⟨finalEnv, ?_, finalReady, ?_⟩
      · apply BlockContinues.range_step
        · have inside := (List.getElem?_eq_some_iff.mp item).1
          simp [Calls.rangeContinues]
          omega
        · simpa only [Int.natCast_add, Int.natCast_one] using visit.append continued
      · cases accepted : data.test left right <;>
          simpa [List.countP_cons, accepted, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using result

/-- Each outer iteration starts its partner scan at the same index, counting each unordered pair once. -/
theorem outer_loop (data : Data) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (valid : Valid data heap) (remaining : List Nat) (position count : Nat) (env : HeapCalls.Env)
    (suffix : data.masks.drop position = remaining) (ready : Ready data env)
    (getCount : env "count" = some (number count)) :
    BlockReturns program env
      [.rangeStepLoop "i" position data.masks.length 1 outerCode [], masks.code[7]'(by decide)]
      heap (number (count + countPairs data.test remaining)) heap globals := by
  induction remaining generalizing position count env with
  | nil =>
      simp only [countPairs, Nat.add_zero]
      apply BlockReturns.range_done
      · have finished := List.drop_eq_nil_iff.mp suffix
        simp [Calls.rangeContinues]
        omega
      · exact BlockReturns.return_value (Evaluates.lookup getCount)
  | cons left rest ih =>
      have item : data.masks[position]? = some left := by
        simpa [suffix] using (List.head?_drop (l := data.masks) (i := position)).symm
      have tail : data.masks.drop (position + 1) = rest := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      let bound := setLocal env "i" (number position)
      have boundReady : Ready data bound := ready.set_auxiliary "i" _ (by decide) (by decide) (by decide)
      have boundCount : bound "count" = some (number count) := by simpa [bound, setLocal] using getCount
      obtain ⟨finalEnv, row, nextReady, nextCount⟩ := row_loop data heap globals valid position left item
        (left :: rest) position count bound suffix boundReady rfl boundCount
      have visited : BlockContinues program bound outerCode heap finalEnv heap globals := by
        apply BlockContinues.for_range
          (.cons (Evaluates.lookup (show bound "i" = some (number position) from rfl))
            (.cons (Evaluates.lookup boundReady.size) .nil)) rfl
        exact row
      apply BlockReturns.range_step
      · have inside := (List.getElem?_eq_some_iff.mp item).1
        simp [Calls.rangeContinues]
        omega
      · apply visited.returns
        simpa only [Int.natCast_add, Int.natCast_one, countPairs, Nat.add_assoc] using
          ih (position + 1) (count + (left :: rest).countP (data.test left)) finalEnv tail nextReady nextCount

/-- The complete nested source loops return the mathematical count from any initial counter. -/
theorem result (data : Data) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (valid : Valid data heap) (count : Nat) (env : HeapCalls.Env) (ready : Ready data env)
    (getCount : env "count" = some (number count)) :
    BlockReturns program env [masks.code[6]'(by decide), masks.code[7]'(by decide)]
      heap (number (count + countPairs data.test data.masks)) heap globals := by
  apply BlockReturns.for_range (.cons (Evaluates.lookup ready.size) .nil) rfl
  exact outer_loop data heap globals valid data.masks 0 count env rfl ready getCount

end solvers.«90».counting
