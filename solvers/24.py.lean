import «.formalsw».solvers.«24.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p24.py»
import lang.py.FormalSW.Permutations
import lang.py.FormalSW.HeapSort
import Mathlib.Data.List.Pairwise
import Mathlib.Data.List.InsertIdx

namespace solvers.«24»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P24Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

namespace PermutationRank

private theorem choices_values (items : List α) :
    (Permutations.choices items).map Prod.fst = items := by
  induction items with
  | nil => rfl
  | cons first rest ih => simp [Permutations.choices, List.map_map, Function.comp_def, ih]

private theorem choice_sublist (selected : (value, remaining) ∈ Permutations.choices items) :
    remaining.Sublist items := by
  induction items generalizing remaining with
  | nil => simp [Permutations.choices] at selected
  | cons first rest ih =>
      simp only [Permutations.choices, List.mem_cons, List.mem_map] at selected
      rcases selected with equal | ⟨⟨other, tail⟩, member, equal⟩
      · cases equal
        exact List.sublist_cons_self _ _
      · cases equal
        exact (ih member).cons_cons first

/-- With a strictly increasing pool, positional permutations are already lexicographic. -/
theorem rows_sorted (length : Nat) (items : List Int) (sorted : items.Pairwise (· < ·)) :
    (Permutations.rows length items).Pairwise (fun a b => HeapSort.lexInt a b = true) := by
  induction length generalizing items with
  | zero => simp [Permutations.rows]
  | succ length ih =>
      rw [Permutations.rows, List.pairwise_flatMap]
      constructor
      · intro pair selected
        have shorter := sorted.sublist (choice_sublist selected)
        apply List.Pairwise.map
        · intro a b less
          simpa [HeapSort.lexInt] using less
        · exact ih pair.2 shorter
      · have choices : (Permutations.choices items).Pairwise (fun a b => a.1 < b.1) := by
          apply List.pairwise_map.mp
          rw [choices_values]
          exact sorted
        apply choices.imp
        intro left right less a aMember b bMember
        obtain ⟨aTail, _, rfl⟩ := List.mem_map.mp aMember
        obtain ⟨bTail, _, rfl⟩ := List.mem_map.mp bMember
        simp [HeapSort.lexInt, Int.ne_of_lt less, less]

private theorem lex_asymm (a b : List Int) (less : HeapSort.lexInt a b = true) :
    HeapSort.lexInt b a = false := by
  induction a generalizing b with
  | nil => cases b <;> simp_all [HeapSort.lexInt]
  | cons first rest ih =>
      cases b with
      | nil => simp [HeapSort.lexInt] at less
      | cons next tail =>
          by_cases same : first = next
          · subst next
            simpa [HeapSort.lexInt] using ih tail (by simpa [HeapSort.lexInt] using less)
          · have order : first < next := by simpa [HeapSort.lexInt, same] using less
            simp [HeapSort.lexInt, Ne.symm same]; omega

private theorem sort_sorted (items : List (List Int))
    (sorted : items.Pairwise (fun a b => HeapSort.lexInt a b = true)) :
    HeapSort.sortBy HeapSort.lexInt items = items := by
  induction items with
  | nil => rfl
  | cons first rest ih =>
      obtain ⟨before, ordered⟩ := List.pairwise_cons.mp sorted
      rw [HeapSort.sortBy, ih ordered]
      cases rest with
      | nil => rfl
      | cons next tail =>
          simp [HeapSort.insertBy, lex_asymm first next (before next List.mem_cons_self)]

theorem sort_rows (items : List Int) (sorted : items.Pairwise (· < ·)) :
    HeapSort.sortBy HeapSort.lexInt (Permutations.rows items.length items) =
      Permutations.rows items.length items := sort_sorted _ (rows_sorted _ _ sorted)


private theorem choices_at (items : List α) (position : Nat) (bound : position < items.length) :
    (Permutations.choices items)[position]? = some (items[position], items.eraseIdx position) := by
  induction items generalizing position with
  | nil => simp at bound
  | cons first rest ih =>
      cases position with
      | zero => rfl
      | succ position =>
          have inside : position < rest.length := by simpa using bound
          simpa [Permutations.choices, List.getElem?_map] using
            congrArg (Option.map (fun pair : α × List α => (pair.1, first :: pair.2))) (ih position inside)

private theorem flatten_at (blocks : List (List α)) (width block offset : Nat)
    (sizes : ∀ row ∈ blocks, row.length = width)
    (blockBound : block < blocks.length) (offsetBound : offset < width) :
    blocks.flatten[block * width + offset]? = blocks[block][offset]? := by
  induction blocks generalizing block with
  | nil => simp at blockBound
  | cons first rest ih =>
      have size := sizes first List.mem_cons_self
      cases block with
      | zero =>
          simpa only [List.flatten_cons, Nat.zero_mul, Nat.zero_add, List.getElem_cons_zero] using
            (List.getElem?_append_left (l₁ := first) (l₂ := rest.flatten) (by omega : offset < first.length))
      | succ block =>
          have bound : block < rest.length := by simpa using blockBound
          have later : first.length ≤ (block + 1) * width + offset := by rw [size, Nat.succ_mul]; omega
          have index : (block + 1) * width + offset - first.length = block * width + offset := by
            rw [size, Nat.succ_mul]; omega
          rw [List.flatten_cons, List.getElem?_append_right later, index]
          exact ih block (fun row member => sizes row (List.mem_cons_of_mem _ member)) bound

private theorem factorial_positive (size : Nat) : 0 < Math.factorial size := by
  induction size with
  | zero => decide
  | succ size ih => exact Nat.mul_pos (by omega) ih

/-- Choose a factorial-sized block, then repeat within the remaining pool. -/
def unrank : Nat → List Int → Nat → List Int
  | 0, _, _ => []
  | size + 1, items, rank =>
      let position := rank / Math.factorial size
      items[position]?.getD 0 :: unrank size (items.eraseIdx position) (rank % Math.factorial size)

/-- The factoradic decomposition selects exactly the matching positional permutation. -/
theorem unrank_at (size : Nat) (items : List Int) (rank : Nat)
    (length : items.length = size) (bound : rank < Math.factorial size) :
    (Permutations.rows size items)[rank]? = some (unrank size items rank) := by
  induction size generalizing items rank with
  | zero =>
      have zero : rank = 0 := by simpa [Math.factorial] using bound
      subst rank
      rfl
  | succ size ih =>
      let width := Math.factorial size
      let position := rank / width
      have positive : 0 < width := factorial_positive size
      have positionBound : position < items.length := by
        apply (Nat.div_lt_iff_lt_mul positive).mpr
        simpa [length, width, Math.factorial] using bound
      have offsetBound : rank % width < width := Nat.mod_lt rank positive
      have choice := choices_at items position positionBound
      let blocks := (Permutations.choices items).map fun pair =>
        (Permutations.rows size pair.2).map (pair.1 :: ·)
      have blockSizes : ∀ row ∈ blocks, row.length = width := by
        intro row member
        obtain ⟨pair, selected, rfl⟩ := List.mem_map.mp member
        have shorter := Permutations.choice_length selected
        have remaining : pair.2.length = size := by omega
        simp [List.length_map, Permutations.rows_length, remaining, Permutations.count_full, width]
      have blockBound : position < blocks.length := by simpa [blocks] using positionBound
      have flat := flatten_at blocks width position (rank % width) blockSizes blockBound offsetBound
      have decomposition : position * width + rank % width = rank := by
        simpa [position, Nat.mul_comm] using Nat.div_add_mod rank width
      rw [decomposition] at flat
      have blockAt : blocks[position] =
          (Permutations.rows size (items.eraseIdx position)).map (items[position] :: ·) := by
        have mapped := congrArg (Option.map (fun pair : Int × List Int =>
          (Permutations.rows size pair.2).map (pair.1 :: ·))) choice
        simpa [blocks, ← List.getElem?_map, List.getElem?_eq_getElem blockBound] using mapped
      have shorter : (items.eraseIdx position).length = size := by
        rw [List.length_eraseIdx_of_lt positionBound, length]
        omega
      have recursive := ih (items.eraseIdx position) (rank % width) shorter offsetBound
      rw [blockAt, List.getElem?_map, recursive] at flat
      simpa [Permutations.rows, List.flatMap, blocks, unrank, position, width,
        List.getElem?_eq_getElem positionBound] using flat

/-- On the optimized solver's domain, sorting and one-based selection give the factoradic result. -/
theorem reference_unrank (items : List Int) (rank : Nat) (sorted : items.Pairwise (· < ·))
    (bound : rank < Math.factorial items.length) :
    data.«project-euler-statements».data.py.project_euler_statements.p24.reference items ((rank : Int) + 1) =
      unrank items.length items rank := by
  rw [data.«project-euler-statements».data.py.project_euler_statements.p24.reference, sort_rows items sorted]
  have length := Permutations.rows_full_length items
  have valid : 0 ≤ (rank : Int) + 1 - 1 ∧ (rank : Int) + 1 - 1 < (Permutations.rows items.length items).length := by
    rw [length]
    omega
  rw [data.«project-euler-statements».data.py.project_euler_statements.p24.selectRow, if_pos valid]
  simp only [Int.add_sub_cancel, Int.toNat_natCast, unrank_at items.length items rank rfl bound, Option.getD_some]

end PermutationRank

namespace Factorials

def table (size : Nat) (value : Nat → Nat) : List Mutable.Value :=
  (List.range size).map fun n => number (value n)

@[simp] theorem table_length (size : Nat) (value : Nat → Nat) : (table size value).length = size := by
  simp [table]

private theorem table_lookup (size : Nat) (value : Nat → Nat) (position : Nat) (inside : position < size) :
    (table size value)[position]? = some (number (value position)) := by
  simp [table, inside]

private theorem table_set (size : Nat) (value : Nat → Nat) (position replacement : Nat) (inside : position < size) :
    (table size value).set position (number replacement) =
      table size (fun n => if n = position then replacement else value n) := by
  apply List.ext_getElem?
  intro n
  by_cases bounded : n < size
  · by_cases same : position = n
    · subst n
      rw [List.getElem?_set_self (by simpa using inside), table_lookup size _ position inside]
      simp
    · rw [List.getElem?_set_ne same, table_lookup size value n bounded, table_lookup size _ n bounded]
      simp [Ne.symm same]
  · have outside : size ≤ n := by omega
    simp only [List.getElem?_eq_none (by simpa using outside : ((table size value).set position (number replacement)).length ≤ n),
      List.getElem?_eq_none (by simpa using outside : (table size (fun k => if k = position then replacement else value k)).length ≤ n)]

private theorem table_congr (size : Nat) (f g : Nat → Nat) (equal : ∀ n, n < size → f n = g n) :
    table size f = table size g := by
  apply List.map_congr_left
  intro n member
  rw [equal n (List.mem_range.mp member)]

def before (next index : Nat) : Nat := if index < next then Math.factorial index else 1

def body : List Stmt :=
  [.assign [.subscript (.name "fact") (.name "i")]
    (.binOp (.subscript (.name "fact") (.binOp (.name "i") .sub (.constant (.int 1)))) .mult (.name "i"))]

private theorem step (size next address : Nat) (lower : 2 ≤ next) (upper : next < size)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (fact : env "fact" = some (.ref address)) (index : env "i" = some (number next))
    (found : heap[address]? = some (.list (table size (before next)))) :
    BlockContinues program env body heap env (heap.set address (.list (table size (before (next + 1))))) := by
  have previous : next - 1 < size := by omega
  have read : Evaluates program env (.subscript (.name "fact") (.binOp (.name "i") .sub (.constant (.int 1))))
      heap (number (Math.factorial (next - 1))) heap := by
    apply Evaluates.subscript (by intros; simp) (Evaluates.lookup fact)
      (Evaluates.binary (Evaluates.lookup index) Evaluates.constant
        (value := number (↑(next - 1 : Nat))) (finalHeap := heap) (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub (by omega : 1 ≤ next)]))
    have item := table_lookup size (before next) (next - 1) previous
    have operation := Mutable.getItem_list heap address (next - 1) (table size (before next)) found (by simpa using previous)
    rw [(List.getElem?_eq_some_iff.mp item).2] at operation
    simpa [before, show next - 1 < next by omega] using operation
  have product : Math.factorial (next - 1) * next = Math.factorial next := by
    cases next with
    | zero => omega
    | succ next => simp [Math.factorial, Nat.mul_comm]
  have calculated : Evaluates program env
      (.binOp (.subscript (.name "fact") (.binOp (.name "i") .sub (.constant (.int 1)))) .mult (.name "i"))
      heap (number (Math.factorial next)) heap := by
    apply Evaluates.binary read (Evaluates.lookup index)
    simp [Mutable.binary, scalarBinary, number, ← product]
  apply BlockContinues.setItem (by intros; simp) calculated
    (.cons (Evaluates.lookup fact) (.cons (Evaluates.lookup index) .nil))
  have updated : (table size (before next)).set next (number (Math.factorial next)) =
      table size (before (next + 1)) := by
    rw [table_set size (before next) next (Math.factorial next) upper]
    apply table_congr
    intro n _
    by_cases equal : n = next
    · simp [equal, before]
    · have same : n < next + 1 ↔ n < next := by omega
      simp [before, equal, same]
  simp [Mutable.setItem, Mutable.index, found, number, show ¬(next : Int) < 0 by omega,
    show ¬ size ≤ next by omega, Except.map, updated]

private def arrayHeap (base : Mutable.Heap) (size next : Nat) : Mutable.Heap :=
  base ++ [.list (table size (before next))]

/-- The source loop fills each factorial once and leaves the other locals untouched. -/
theorem fill (maximum next : Nat) (lower : 2 ≤ next) (env : HeapCalls.Env) (base : Mutable.Heap)
    (fact : env "fact" = some (.ref base.length)) :
    ∃ finalEnv, (∀ name, name ≠ "i" → finalEnv name = env name) ∧
      BlockContinues program env [.rangeStepLoop "i" next (maximum + 1) 1 body []]
        (arrayHeap base (maximum + 1) next) finalEnv
        (base ++ [.list (table (maximum + 1) Math.factorial)]) := by
  by_cases active : next < maximum + 1
  · let bound := setLocal env "i" (number next)
    obtain ⟨finalEnv, preserved, finished⟩ := fill maximum (next + 1) (by omega) bound base
      (by simpa [bound, setLocal] using fact)
    refine ⟨finalEnv, ?_, ?_⟩
    · intro name other
      rw [preserved name other]
      simp [bound, setLocal, other]
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      have written := step (maximum + 1) next base.length lower active bound
        (arrayHeap base (maximum + 1) next) (by simpa [bound, setLocal] using fact) rfl (by simp [arrayHeap])
      have advanced : (arrayHeap base (maximum + 1) next).set base.length
          (.list (table (maximum + 1) (before (next + 1)))) = arrayHeap base (maximum + 1) (next + 1) := by
        simp [arrayHeap, List.set_append_right]
      rw [advanced] at written
      simpa [body, Calls.loopContinuation, Calls.loopControl] using written.append finished
  · have complete : table (maximum + 1) (before next) = table (maximum + 1) Math.factorial := by
      apply table_congr
      intro n bound
      simp [before, show n < next by omega]
    refine ⟨env, fun _ _ => rfl, ?_⟩
    rw [arrayHeap, complete]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
termination_by maximum + 1 - next


private theorem initial_value (index : Nat) : before 2 index = 1 := by
  cases index with
  | zero => rfl
  | succ index =>
      cases index with
      | zero => rfl
      | succ index => simp [before]; omega

private theorem initial_table (size : Nat) : table size (before 2) = List.replicate size (number 1) := by
  simp [table, initial_value]

def initialExpression := Expr.binOp (.list [.constant (.int 1)]) .mult
  (.binOp (.name "k") .add (.constant (.int 1)))

private theorem initialize_list (maximum : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (bound : env "k" = some (number maximum)) :
    Evaluates program env initialExpression heap (.ref (heap.length + 1))
      (arrayHeap (heap ++ [.list [number 1]]) (maximum + 1) 2) := by
  let base := heap ++ [.list [number 1]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number (maximum + 1))
    (middle := base) (afterRight := base)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · apply Evaluates.binary (Evaluates.lookup bound) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, base, arrayHeap, initial_table, Except.map]

def setupBody : List Stmt :=
  [.assign [.name "k"] (.builtinCall .len [.name "digits"]),
   .assign [.name "fact"] initialExpression,
   .forIn (.name "i") (.call (.name "range") [.constant (.int 2), .binOp (.name "k") .add (.constant (.int 1))]) body []]

/-- The actual precomputation prefix allocates and fills the complete factorial table. -/
theorem setup (digits : List Int) (address : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (lookup : env "digits" = some (.ref address))
    (found : heap[address]? = some (.list (digits.map fun n => number n))) :
    ∃ finalEnv : HeapCalls.Env,
      finalEnv "k" = some (number digits.length) ∧
      finalEnv "fact" = some (.ref (heap.length + 1)) ∧
      (∀ name, name ≠ "k" → name ≠ "fact" → name ≠ "i" → finalEnv name = env name) ∧
      BlockContinues program env setupBody heap finalEnv
        (heap ++ [.list [number 1], .list (table (digits.length + 1) Math.factorial)]) := by
  let afterLength := setLocal env "k" (number digits.length)
  let afterAllocation := setLocal afterLength "fact" (.ref (heap.length + 1))
  let base := heap ++ [.list [number 1]]
  obtain ⟨finalEnv, preserved, filled⟩ := fill digits.length 2 (by decide) afterAllocation base (by simp [base, afterAllocation, setLocal])
  refine ⟨finalEnv, ?_, ?_, ?_, ?_⟩
  · rw [preserved "k" (by decide)]
    simp [afterAllocation, afterLength, setLocal]
  · rw [preserved "fact" (by decide)]
    simp [afterAllocation, setLocal]
  · intro name notK notFact notI
    rw [preserved name notI]
    simp [afterAllocation, afterLength, setLocal, notK, notFact]
  · have counted : BlockContinues program env [.assign [.name "k"] (.builtinCall .len [.name "digits"])]
        heap afterLength heap := BlockContinues.assign
          (Evaluates.builtin (.cons (Evaluates.lookup lookup) .nil) (by simp [primitive, Mutable.length, found, number]))
    have allocated := BlockContinues.assign (name := "fact") (initialize_list digits.length afterLength heap rfl)
    have loop : BlockContinues program afterAllocation
        [.forIn (.name "i") (.call (.name "range") [.constant (.int 2), .binOp (.name "k") .add (.constant (.int 1))]) body []]
        (arrayHeap base (digits.length + 1) 2) finalEnv
        (base ++ [.list (table (digits.length + 1) Math.factorial)]) := by
      have nextBound : Evaluates program afterAllocation (.binOp (.name "k") .add (.constant (.int 1)))
          (arrayHeap base (digits.length + 1) 2) (number (digits.length + 1)) (arrayHeap base (digits.length + 1) 2) := by
        apply Evaluates.binary (av := number digits.length)
          (Evaluates.lookup (name := "k") (by simp [afterAllocation, afterLength, setLocal])) Evaluates.constant
        simp [Mutable.binary, scalarBinary, number]
      exact BlockContinues.for_range (.cons Evaluates.constant (.cons nextBound .nil)) rfl filled
    simpa only [setupBody, base, List.append_assoc, List.cons_append, List.nil_append] using
      counted.append (allocated.append loop)

theorem source_prefix :
    (executableBody ((findFunctionIn program.body "nth_lexicographic_permutation").getD ([], [])).2).take 3 =
      setupBody := rfl

end Factorials

namespace Selection

private def numbers (items : List Int) : List Mutable.Value := items.map number
private def texts (items : List Int) : List Mutable.Value := items.map fun n => .scalar (.string (toString n))

def workHeap (base : Mutable.Heap) (available output : List Int) : Mutable.Heap :=
  base ++ [.list (numbers available), .list (texts output)]

private abbrev appendExpression := Expr.call (.attribute (.name "out") "append")
  [.builtinCall .str [.call (.attribute (.name "available") "pop") [.name "idx"]]]

/-- The source pop/str/append expression moves the selected value between private lists. -/
private theorem append_popped (base : Mutable.Heap) (available output : List Int) (position : Nat)
    (bound : position < available.length) (env : HeapCalls.Env)
    (getAvailable : env "available" = some (.ref base.length))
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (getIndex : env "idx" = some (number position)) :
    Evaluates program env appendExpression (workHeap base available output) (.scalar .none)
      (workHeap base (available.eraseIdx position) (output ++ [available[position]])) := by
  have availableFound : (workHeap base available output)[base.length]? = some (.list (numbers available)) := by
    simp [workHeap]
  have popped : Evaluates program env (.call (.attribute (.name "available") "pop") [.name "idx"])
      (workHeap base available output) (number available[position]) (workHeap base (available.eraseIdx position) output) := by
    apply Evaluates.pop getAvailable
      (by simp [bindMethod, availableFound]) (.cons (Evaluates.lookup getIndex) .nil)
    have operation := Mutable.popItem_list (workHeap base available output) base.length position (numbers available)
      availableFound (by simpa [numbers] using bound)
    simpa [primitive, workHeap, numbers, List.eraseIdx_map, List.set_append_right] using operation
  have converted : Evaluates program env
      (.builtinCall .str [.call (.attribute (.name "available") "pop") [.name "idx"]])
      (workHeap base available output) (.scalar (.string (toString available[position])))
      (workHeap base (available.eraseIdx position) output) := by
    exact Evaluates.builtin (.cons popped .nil) rfl
  have appended := Evaluates.append (Evaluates.lookup getOutput)
    (by simp [workHeap] : (workHeap base available output)[base.length + 1]? = some (.list (texts output))) converted
    (by simp [workHeap] : (workHeap base (available.eraseIdx position) output)[base.length + 1]? = some (.list (texts output)))
  simpa [workHeap, List.set_append_right, texts] using appended

def body : List Stmt :=
  [.assign [.name "block"] (.subscript (.name "fact") (.binOp (.name "i") .sub (.constant (.int 1)))),
   .assign [.name "idx"] (.binOp (.name "n") .floorDiv (.name "block")),
   .augAssign (.name "n") .mod (.name "block"),
   .expr appendExpression]

theorem source_body :
    (match ((executableBody ((findFunctionIn program.body "nth_lexicographic_permutation").getD ([], [])).2)[6]? : Option Stmt) with
     | some (.forIn _ _ body _) => body
     | _ => []) = body := rfl


def nextEnv (env : HeapCalls.Env) (size rank : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "block" (number (Math.factorial size)))
    "idx" (number (rank / Math.factorial size : Nat))) "n" (number (rank % Math.factorial size : Nat))

private theorem step (base : Mutable.Heap) (available output : List Int) (maximum size rank factAddress : Nat)
    (length : available.length = size + 1) (rankBound : rank < Math.factorial (size + 1)) (sizeBound : size ≤ maximum)
    (found : base[factAddress]? = some (.list (Factorials.table (maximum + 1) Math.factorial)))
    (env : HeapCalls.Env)
    (getFact : env "fact" = some (.ref factAddress))
    (getAvailable : env "available" = some (.ref base.length))
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (getRank : env "n" = some (number rank)) (getSize : env "i" = some (number (size + 1))) :
    BlockContinues program env body (workHeap base available output) (nextEnv env size rank)
      (workHeap base (available.eraseIdx (rank / Math.factorial size))
        (output ++ [available[rank / Math.factorial size]'(by
          apply (Nat.div_lt_iff_lt_mul (PermutationRank.factorial_positive size)).mpr
          simpa [Math.factorial, length] using rankBound)])) := by
  let position := rank / Math.factorial size
  have positive := PermutationRank.factorial_positive size
  have positionBound : position < available.length := by
    apply (Nat.div_lt_iff_lt_mul positive).mpr
    simpa [Math.factorial, length] using rankBound
  have factBound : factAddress < base.length := by
    by_cases valid : factAddress < base.length
    · exact valid
    · rw [List.getElem?_eq_none (by omega)] at found
      cases found
  have tableFound : (workHeap base available output)[factAddress]? = some (.list (Factorials.table (maximum + 1) Math.factorial)) := by
    simpa [workHeap, List.getElem?_append_left factBound] using found
  let withBlock := setLocal env "block" (number (Math.factorial size))
  let withIndex := setLocal withBlock "idx" (number position)
  have read : Evaluates program env (.subscript (.name "fact") (.binOp (.name "i") .sub (.constant (.int 1))))
      (workHeap base available output) (number (Math.factorial size)) (workHeap base available output) := by
    apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getFact)
      (Evaluates.binary (Evaluates.lookup getSize) Evaluates.constant (value := number size)
        (finalHeap := workHeap base available output) (by simp [number, Mutable.binary, scalarBinary]))
    have item := Factorials.table_lookup (maximum + 1) Math.factorial size (by omega)
    have operation := Mutable.getItem_list (workHeap base available output) factAddress size
      (Factorials.table (maximum + 1) Math.factorial) tableFound (by simp; omega)
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using operation
  have divided : Evaluates program withBlock (.binOp (.name "n") .floorDiv (.name "block"))
      (workHeap base available output) (number position) (workHeap base available output) := by
    apply Evaluates.binary (av := number rank) (bv := number (Math.factorial size))
    · exact Evaluates.lookup (by simpa [withBlock, setLocal] using getRank)
    · exact Evaluates.lookup rfl
    · simp [Mutable.binary, scalarBinary, number, positive, Nat.ne_of_gt positive, position]
  have reduced : BlockContinues program withIndex [.augAssign (.name "n") .mod (.name "block")]
      (workHeap base available output) (nextEnv env size rank) (workHeap base available output) := by
    apply BlockContinues.augAssign (left := number rank)
      (by simpa [withIndex, withBlock, setLocal] using getRank) (Evaluates.lookup (name := "block") (value := number (Math.factorial size)) (by simp [withIndex, withBlock, setLocal]))
    simp [inplace, Mutable.binary, scalarBinary, number, positive, Nat.ne_of_gt positive]
  have moved := append_popped base available output position positionBound (nextEnv env size rank)
    (by simpa [nextEnv, setLocal] using getAvailable) (by simpa [nextEnv, setLocal] using getOutput)
    (by simp [nextEnv, setLocal, position])
  simpa only [body, List.cons_append, List.nil_append] using
    (BlockContinues.assign (name := "block") read).append
      ((BlockContinues.assign (name := "idx") divided).append (reduced.append (BlockContinues.discard moved)))


/-- Every iteration consumes one pool entry and extends the output by the factoradic choice. -/
theorem loop (size : Nat) (base : Mutable.Heap) (available output : List Int) (maximum rank factAddress : Nat)
    (length : available.length = size) (rankBound : rank < Math.factorial size) (sizeBound : size ≤ maximum)
    (found : base[factAddress]? = some (.list (Factorials.table (maximum + 1) Math.factorial)))
    (env : HeapCalls.Env) (getFact : env "fact" = some (.ref factAddress))
    (getAvailable : env "available" = some (.ref base.length))
    (getOutput : env "out" = some (.ref (base.length + 1))) (getRank : env "n" = some (number rank)) :
    ∃ finalEnv : HeapCalls.Env, finalEnv "out" = some (.ref (base.length + 1)) ∧
      BlockContinues program env [.rangeStepLoop "i" size 0 (-1) body []]
        (workHeap base available output) finalEnv
        (workHeap base [] (output ++ PermutationRank.unrank size available rank)) := by
  induction size generalizing available output rank env with
  | zero =>
      have empty := List.eq_nil_of_length_eq_zero length
      subst available
      refine ⟨env, getOutput, ?_⟩
      simpa [PermutationRank.unrank] using
        (BlockContinues.range_done (module := program) (env := env) (name := "i") (body := body)
          (heap := workHeap base [] output) (by decide : Calls.rangeContinues 0 0 (-1) = false) BlockContinues.nil)
  | succ size ih =>
      let position := rank / Math.factorial size
      have positive := PermutationRank.factorial_positive size
      have positionBound : position < available.length := by
        apply (Nat.div_lt_iff_lt_mul positive).mpr
        simpa [Math.factorial, length] using rankBound
      have remainingLength : (available.eraseIdx position).length = size := by
        rw [List.length_eraseIdx_of_lt positionBound, length]
        omega
      let bound := setLocal env "i" (number (size + 1))
      have executed := step base available output maximum size rank factAddress length rankBound (by omega) found bound
        (by simpa [bound, setLocal] using getFact) (by simpa [bound, setLocal] using getAvailable)
        (by simpa [bound, setLocal] using getOutput) (by simpa [bound, setLocal] using getRank) rfl
      obtain ⟨finalEnv, finalOutput, finished⟩ := ih (available.eraseIdx position) (output ++ [available[position]])
        (rank % Math.factorial size) remainingLength (Nat.mod_lt rank positive) (by omega) (nextEnv bound size rank)
        (by simpa [nextEnv, bound, setLocal] using getFact)
        (by simpa [nextEnv, bound, setLocal] using getAvailable)
        (by simpa [nextEnv, bound, setLocal] using getOutput) (by simp [nextEnv, setLocal])
      refine ⟨finalEnv, finalOutput, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues])
      simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, PermutationRank.unrank, position, bound, number, Int.add_assoc,
        List.getElem?_eq_getElem positionBound, List.append_assoc] using executed.append finished

end Selection

private abbrev solverArgs : List Arg :=
  [⟨"digits", some (.subscript (.name "list") (.name "int")), none⟩,
   ⟨"n", some (.name "int"), none⟩]

/-- The original optimized Python returns the factoradic permutation and preserves the input heap. -/
theorem optimized.result (heap : Mutable.Heap) (address : Nat) (digits : List Int) (rank : Nat)
    (found : heap[address]? = some (.list (digits.map number)))
    (rankBound : rank < Math.factorial digits.length) :
    ∃ extra, ReturnsValue program "nth_lexicographic_permutation" [.ref address, number (rank + 1)] heap
      (.scalar (.string (String.intercalate "" ((PermutationRank.unrank digits.length digits rank).map toString))))
      (heap ++ extra) := by
  let entry := argumentEnv solverArgs [.ref address, number (rank + 1)]
  obtain ⟨afterFacts, getLength, getFact, preserved, setup⟩ := Factorials.setup digits address entry heap rfl found
  let base := heap ++ [.list [number 1], .list (Factorials.table (digits.length + 1) Math.factorial)]
  let afterRank := setLocal afterFacts "n" (number rank)
  let afterCopy := setLocal afterRank "available" (.ref base.length)
  let ready := setLocal afterCopy "out" (.ref (base.length + 1))
  have original : base[address]? = some (.list (digits.map number)) := by
    have inside : address < heap.length := by
      by_cases valid : address < heap.length
      · exact valid
      · rw [List.getElem?_eq_none (by omega)] at found
        cases found
    simpa [base, List.getElem?_append_left inside] using found
  have reset : BlockContinues program afterFacts [.augAssign (.name "n") .sub (.constant (.int 1))]
      base afterRank base := by
    apply BlockContinues.augAssign (left := number (rank + 1))
    · rw [preserved "n" (by decide) (by decide) (by decide)]
      rfl
    · exact Evaluates.constant
    · simp [inplace, Mutable.binary, scalarBinary, number]
  have copied : Evaluates program afterRank (.subscript (.name "digits") (.slice none none none))
      base (.ref base.length) (base ++ [.list (digits.map number)]) := by
    have digitsLookup : afterRank "digits" = some (.ref address) := by
      rw [show afterRank "digits" = afterFacts "digits" from rfl,
        preserved "digits" (by decide) (by decide) (by decide)]
      rfl
    exact Evaluates.slice (.cons (Evaluates.lookup digitsLookup)
      (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
      (Mutable.getSlice_list_copy base address _ original)
  have allocated : BlockContinues program afterCopy
      [.annAssign (.name "out") (.subscript (.name "list") (.name "str")) (some (.list []))]
      (base ++ [.list (digits.map number)]) ready (Selection.workHeap base digits []) := by
    have allocation := BlockContinues.annAssign (globals := []) (module := program) (env := afterCopy) (name := "out")
      (annotation := .subscript (.name "list") (.name "str"))
      (Evaluates.list (heap := base ++ [.list (digits.map number)]) .nil)
    simpa [Selection.workHeap, Selection.numbers, Selection.texts, ready, List.append_assoc] using allocation
  obtain ⟨finalEnv, finalOutput, executed⟩ := Selection.loop digits.length base digits [] digits.length rank (heap.length + 1)
    rfl rankBound (by omega) (by simp [base]) ready
    (by simpa [ready, afterCopy, afterRank, setLocal] using getFact)
    (by simp [ready, afterCopy, setLocal]) (by simp [ready, setLocal]) (by simp [ready, afterCopy, afterRank, setLocal])
  have loop : BlockContinues program ready
      [.forIn (.name "i") (.call (.name "range") [.name "k", .constant (.int 0), .unaryOp .uSub (.constant (.int 1))]) Selection.body []]
      (Selection.workHeap base digits []) finalEnv
      (Selection.workHeap base [] (PermutationRank.unrank digits.length digits rank)) := by
    apply BlockContinues.for_range (values := [number digits.length, number 0, number (-1)])
    · exact .cons (Evaluates.lookup (by simpa [ready, afterCopy, afterRank, setLocal] using getLength))
        (.cons Evaluates.constant (.cons (Evaluates.unary Evaluates.constant rfl) .nil))
    · rfl
    · simpa using executed
  have joined : Evaluates program finalEnv (.call (.attribute (.constant (.string "")) "join") [.name "out"])
      (Selection.workHeap base [] (PermutationRank.unrank digits.length digits rank))
      (.scalar (.string (String.intercalate "" ((PermutationRank.unrank digits.length digits rank).map toString))))
      (Selection.workHeap base [] (PermutationRank.unrank digits.length digits rank)) := by
    apply Evaluates.join Evaluates.constant (.cons (Evaluates.lookup finalOutput) .nil)
    · exact primitive_join_list _ (base.length + 1) "" _ (by simp [Selection.workHeap, Selection.texts, List.map_map])
    · intro name impossible
      cases impossible
  refine ⟨[.list [number 1], .list (Factorials.table (digits.length + 1) Math.factorial),
    .list [], .list (Selection.texts (PermutationRank.unrank digits.length digits rank))], ?_⟩
  apply BlockReturns.function rfl rfl
  have completed := (setup.append (reset.append ((BlockContinues.assign (name := "available") copied).append
    (allocated.append loop)))).returns (BlockReturns.return_value joined (rest := []))
  simpa [Factorials.setupBody, Selection.workHeap, Selection.numbers, base, List.append_assoc] using completed

/-- Source-level equivalence on strictly increasing inputs and valid one-based ranks. -/
theorem optimized.equiv (heap : Mutable.Heap) (address : Nat) (digits : List Int) (rank : Nat)
    (valid : heap.Valid) (found : heap[address]? = some (.list (digits.map number)))
    (sorted : digits.Pairwise (· < ·)) (rankBound : rank < Math.factorial digits.length) :
    (HeapCalls.call program "nth_lexicographic_permutation" [.ref address, number (rank + 1)] heap).Equiv
      (HeapCalls.call P24Ast.module "naive_text" [.ref address, number (rank + 1)] heap) := by
  obtain ⟨extra, executed⟩ := optimized.result heap address digits rank found rankBound
  obtain ⟨referenceExtra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p24.naive_text.exact_result
      heap address digits ((rank : Int) + 1) found
  rw [PermutationRank.reference_unrank digits rank sorted rankBound] at reference
  exact Mutable.Call.equiv_of_finishes executed.finishes reference.finishes
    (Mutable.Completion.equivalent_allocations heap extra referenceExtra _ [] valid trivial (by simp))


namespace Domain

def increasing : List Int → Bool
  | a :: b :: rest => decide (a < b) && increasing (b :: rest)
  | _ => true

theorem increasing_iff (items : List Int) : increasing items = true ↔ items.Pairwise (· < ·) := by
  induction items with
  | nil => simp [increasing]
  | cons first rest ih =>
      cases rest with
      | nil => simp [increasing]
      | cons second rest =>
          rw [increasing, Bool.and_eq_true, decide_eq_true_eq, ih]
          constructor
          · rintro ⟨before, sorted⟩
            apply List.pairwise_cons.mpr
            refine ⟨?_, sorted⟩
            intro item member
            rcases List.mem_cons.mp member with rfl | member
            · exact before
            · exact Int.lt_trans before ((List.pairwise_cons.mp sorted).1 item member)
          · intro sorted
            have head := List.pairwise_cons.mp sorted
            exact ⟨head.1 second List.mem_cons_self, head.2⟩

private abbrev orderTest := Expr.compare
  (.subscript (.name "digits") (.binOp (.name "i") .sub (.constant (.int 1))))
  [(.gtE, .subscript (.name "digits") (.name "i"))]
private abbrev orderBody : List Stmt := [.ifThen orderTest [.return (.constant (.bool false))] []]

private theorem order_test (digits : List Int) (heap : Mutable.Heap) (address offset : Nat) (first second : Int)
    (found : heap[address]? = some (.list (digits.map number)))
    (firstAt : digits[offset]? = some first) (secondAt : digits[offset + 1]? = some second)
    (env : HeapCalls.Env) (lookup : env "digits" = some (.ref address)) (index : env "i" = some (number (offset + 1))) :
    Evaluates program env orderTest heap (.scalar (.bool (decide (second ≤ first)))) heap := by
  have read (position : Nat) (value : Int) (elementAt : digits[position]? = some value)
      (expression : Expr) (ordinary : ∀ a b c, expression ≠ .slice a b c)
      (evaluated : Evaluates program env expression heap (number position) heap) :
      Evaluates program env (.subscript (.name "digits") expression) heap (number value) heap := by
    have inside := (List.getElem?_eq_some_iff.mp elementAt).1
    apply Evaluates.subscript ordinary (Evaluates.lookup lookup) evaluated
    have operation := Mutable.getItem_list heap address position (digits.map number) found (by simpa using inside)
    simpa only [List.getElem_map, (List.getElem?_eq_some_iff.mp elementAt).2] using operation
  apply Evaluates.compare
    (read offset first firstAt _ (by intros; simp)
      (Evaluates.binary (Evaluates.lookup index) Evaluates.constant (by simp [number, Mutable.binary, scalarBinary])))
    (read (offset + 1) second secondAt _ (by intros; simp) (Evaluates.lookup index))
  rfl

/-- Adjacent comparisons can reject early; accepted inputs run the supplied continuation. -/
private theorem order_loop (digits remaining : List Int) (heap : Mutable.Heap) (address offset : Nat)
    (found : heap[address]? = some (.list (digits.map number))) (suffix : digits.drop offset = remaining)
    (env : HeapCalls.Env) (lookup : env "digits" = some (.ref address))
    (following : List Stmt) (flag : Bool)
    (rest : ∀ nextEnv, (∀ name, name ≠ "i" → nextEnv name = env name) →
      BlockReturns program nextEnv following heap (.scalar (.bool flag)) heap) :
    BlockReturns program env (.rangeStepLoop "i" (offset + 1) digits.length 1 orderBody [] :: following) heap
      (.scalar (.bool (increasing remaining && flag))) heap := by
  induction remaining using increasing.induct generalizing offset env with
  | case1 first second remaining ih =>
      have size := congrArg List.length suffix
      simp only [List.length_drop, List.length_cons] at size
      have active : offset + 1 < digits.length := by omega
      have firstAt : digits[offset]? = some first := by
        simpa [suffix] using (List.head?_drop (l := digits) (i := offset)).symm
      have tail : digits.drop (offset + 1) = second :: remaining := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      have secondAt : digits[offset + 1]? = some second := by
        simpa [tail] using (List.head?_drop (l := digits) (i := offset + 1)).symm
      let bound := setLocal env "i" (number (offset + 1))
      have checked := order_test digits heap address offset first second found firstAt secondAt bound
        (by simpa [bound, setLocal] using lookup) rfl
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      by_cases accepted : first < second
      · have notRejected : ¬ second ≤ first := by omega
        simp only [increasing, accepted, decide_true, Bool.true_and]
        apply BlockReturns.branch checked (flag := false)
          (by simp [Mutable.truth, scalarTruth, notRejected])
        have continued := ih (offset + 1) tail bound (by simpa [bound, setLocal] using lookup)
          (fun nextEnv preserved => rest nextEnv (fun name other => by rw [preserved name other]; simp [bound, setLocal, other]))
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, orderBody, bound, number, Int.natCast_add] using continued
      · have rejected : second ≤ first := by omega
        simp only [increasing, accepted, decide_false, Bool.false_and]
        apply BlockReturns.branch checked (flag := true)
          (by simp [Mutable.truth, scalarTruth, rejected])
        exact BlockReturns.return_value Evaluates.constant
  | case2 remaining noPair =>
      have small : remaining.length ≤ 1 := by
        cases remaining with
        | nil => simp
        | cons first tail =>
            cases tail with
            | nil => simp
            | cons second tail => exact False.elim (noPair first second tail rfl)
      have size := congrArg List.length suffix
      simp only [List.length_drop] at size
      have finished : digits.length ≤ offset + 1 := by omega
      have accepted : increasing remaining = true := by
        cases remaining with
        | nil => rfl
        | cons first tail =>
            cases tail with
            | nil => rfl
            | cons second tail => simp at small
      rw [accepted, Bool.true_and]
      exact BlockReturns.range_done (by simp [Calls.rangeContinues]; omega) (rest env (fun _ _ => rfl))


private abbrev countBody : List Stmt := [.augAssign (.name "count") .mult (.name "i")]
private abbrev rankTest := Expr.compare (.constant (.int 1)) [(.ltE, .name "n"), (.ltE, .name "count")]
private abbrev countTail : List Stmt :=
  [.assign [.name "count"] (.constant (.int 1)),
   .forIn (.name "i") (.call (.name "range") [.constant (.int 2), .binOp (.builtinCall .len [.name "digits"]) .add (.constant (.int 1))]) countBody [],
   .return rankTest]

private theorem count_loop (maximum next : Nat) (lower : 2 ≤ next) (upper : next ≤ maximum + 1)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (count : env "count" = some (number (Math.factorial (next - 1)))) :
    ∃ finalEnv : HeapCalls.Env, finalEnv "count" = some (number (Math.factorial maximum)) ∧
      (∀ name, name ≠ "i" → name ≠ "count" → finalEnv name = env name) ∧
      BlockContinues program env [.rangeStepLoop "i" next (maximum + 1) 1 countBody []] heap finalEnv heap := by
  by_cases active : next < maximum + 1
  · let bound := setLocal env "i" (number next)
    let advanced := setLocal bound "count" (number (Math.factorial next))
    have product : Math.factorial (next - 1) * next = Math.factorial next := by
      cases next with
      | zero => omega
      | succ next => simp [Math.factorial, Nat.mul_comm]
    have multiplied : BlockContinues program bound countBody heap advanced heap := by
      apply BlockContinues.augAssign (left := number (Math.factorial (next - 1)))
        (by simpa [bound, setLocal] using count) (Evaluates.lookup (name := "i") rfl)
      simp [inplace, Mutable.binary, scalarBinary, number, ← product]
    obtain ⟨finalEnv, finalCount, preserved, finished⟩ := count_loop maximum (next + 1) (by omega) (by omega) heap advanced
      (by simp [advanced, setLocal])
    refine ⟨finalEnv, finalCount, ?_, ?_⟩
    · intro name notI notCount
      rw [preserved name notI notCount]
      simp [advanced, bound, setLocal, notI, notCount]
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa [countBody, Calls.loopContinuation, Calls.loopControl, bound, number] using multiplied.append finished
  · have last : next - 1 = maximum := by omega
    refine ⟨env, by simpa [last] using count, fun _ _ _ => rfl, ?_⟩
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
termination_by maximum + 1 - next

private theorem rank_test (n : Int) (maximum : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) (getCount : env "count" = some (number (Math.factorial maximum))) :
    Evaluates program env rankTest heap (.scalar (.bool (decide (1 ≤ n ∧ n ≤ Math.factorial maximum)))) heap := by
  apply Evaluates.chain Evaluates.constant
  by_cases lower : 1 ≤ n
  · apply Evaluates.Compares.next (Evaluates.lookup getN) (compared := .scalar (.bool true)) (afterComparison := heap)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, lower]
    · rfl
    · exact Evaluates.Compares.last (Evaluates.lookup getCount) (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, lower])
  · simp only [lower, false_and, decide_false]
    exact Evaluates.Compares.stop (Evaluates.lookup getN)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, lower]) rfl

private theorem count_result (digits : List Int) (n : Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (digits.map number))) (env : HeapCalls.Env)
    (lookup : env "digits" = some (.ref address)) (getN : env "n" = some (number n)) :
    BlockReturns program env countTail heap
      (.scalar (.bool (decide (1 ≤ n ∧ n ≤ Math.factorial digits.length)))) heap := by
  let initialized := setLocal env "count" (number 1)
  have len : Evaluates program initialized (.builtinCall .len [.name "digits"]) heap (number digits.length) heap := by
    exact Evaluates.builtin (.cons (Evaluates.lookup (by simpa [initialized, setLocal] using lookup)) .nil)
      (by simp [primitive, Mutable.length, found, number])
  have stop : Evaluates program initialized (.binOp (.builtinCall .len [.name "digits"]) .add (.constant (.int 1)))
      heap (number (digits.length + 1)) heap := by
    apply Evaluates.binary len Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  have counted : ∃ finalEnv : HeapCalls.Env,
      finalEnv "count" = some (number (Math.factorial digits.length)) ∧ finalEnv "n" = some (number n) ∧
      BlockContinues program initialized [.rangeStepLoop "i" 2 (digits.length + 1) 1 countBody []] heap finalEnv heap := by
    by_cases empty : digits.length = 0
    · refine ⟨initialized, by simp [initialized, setLocal, empty, Math.factorial], ?_, ?_⟩
      · simpa [initialized, setLocal] using getN
      · exact BlockContinues.range_done (by simp [Calls.rangeContinues, empty]) .nil
    · obtain ⟨finalEnv, finalCount, preserved, finished⟩ := count_loop digits.length 2 (by decide) (by omega) heap initialized rfl
      exact ⟨finalEnv, finalCount, by rw [preserved "n" (by decide) (by decide)]; simpa [initialized, setLocal] using getN, finished⟩
  obtain ⟨finalEnv, finalCount, finalN, finished⟩ := counted
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range (.cons Evaluates.constant (.cons stop .nil)) rfl
  exact finished.returns (BlockReturns.return_value (rank_test n digits.length heap finalEnv finalN finalCount))

end Domain

/-- The Python prerequisite always returns and checks both strict order and factorial rank bounds. -/
theorem valid_permutation_input.result (digits : List Int) (n : Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (digits.map number))) :
    ReturnsValue program "valid_permutation_input" [.ref address, number n] heap
      (.scalar (.bool (Domain.increasing digits && decide (1 ≤ n ∧ n ≤ Math.factorial digits.length)))) heap := by
  let env := argumentEnv solverArgs [.ref address, number n]
  have len : Evaluates program env (.builtinCall .len [.name "digits"]) heap (number digits.length) heap :=
    Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil) (by simp [primitive, Mutable.length, found, number])
  apply BlockReturns.function rfl rfl
  apply BlockReturns.for_range (.cons Evaluates.constant (.cons len .nil)) rfl
  apply Domain.order_loop digits digits heap address 0 found rfl env rfl Domain.countTail
    (decide (1 ≤ n ∧ n ≤ Math.factorial digits.length))
  intro finalEnv preserved
  exact Domain.count_result digits n heap address found finalEnv
    (by rw [preserved "digits" (by decide)]; rfl) (by rw [preserved "n" (by decide)]; rfl)



private abbrev inputEnv (address : Nat) (n : Int) : HeapCalls.Env :=
  fun name => if name = "digits" then some (.ref address) else if name = "n" then some (number n) else none

private theorem condition_result (digits : List Int) (n : Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (digits.map number))) :
    Evaluates program (inputEnv address n) Solvers.P24Ast.requires_nth_lexicographic_permutation_0 heap
      (.scalar (.bool (Domain.increasing digits && decide (1 ≤ n ∧ n ≤ Math.factorial digits.length)))) heap :=
  Evaluates.call (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup rfl) .nil))
    (valid_permutation_input.result digits n heap address found)

private theorem condition_domain (digits : List Int) (n : Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (digits.map number)))
    (requires : HeapCalls.Condition program (inputEnv address n) Solvers.P24Ast.requires_nth_lexicographic_permutation_0 heap) :
    digits.Pairwise (· < ·) ∧ 1 ≤ n ∧ n ≤ Math.factorial digits.length := by
  have accepted := (condition_result digits n heap address found).condition_iff.mp requires
  have accepted : Domain.increasing digits = true ∧ 1 ≤ n ∧ n ≤ Math.factorial digits.length := by
    simpa only [Mutable.Value.scalar.injEq, Constant.bool.injEq, Bool.and_eq_true, decide_eq_true_eq] using accepted
  exact ⟨(Domain.increasing_iff digits).mp accepted.1, accepted.2⟩

theorem requires_nth_lexicographic_permutation_0_supported
    (heap : Mutable.Heap) (address : Nat) (n : Int) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]") :
    (HeapCalls.evaluate program (inputEnv address n) Solvers.P24Ast.requires_nth_lexicographic_permutation_0 heap).Supported := by
  obtain ⟨digits, found⟩ := Mutable.Value.ints_of_hasType typed
  exact Execution.supported_of_finishes (condition_result digits n heap address found).finishes

theorem nth_lexicographic_permutation.«Equiv(naive)»
    (heap : Mutable.Heap) (address : Nat) (n : Int) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]")
    (requires : HeapCalls.Condition program (inputEnv address n) Solvers.P24Ast.requires_nth_lexicographic_permutation_0 heap) :
    (HeapCalls.call program "nth_lexicographic_permutation" [.ref address, number n] heap).Equiv
      (HeapCalls.call P24Ast.module "naive_text" [.ref address, number n] heap) := by
  obtain ⟨digits, found⟩ := Mutable.Value.ints_of_hasType typed
  obtain ⟨sorted, lower, upper⟩ := condition_domain digits n heap address found requires
  let rank := (n - 1).toNat
  have index : (rank : Int) + 1 = n := by dsimp [rank]; omega
  have bound : rank < Math.factorial digits.length := by omega
  simpa only [index] using optimized.equiv heap address digits rank valid found sorted bound

theorem nth_lexicographic_permutation.«Supported» :
    (∀ (heap : Mutable.Heap) (address : Nat) (n : Int), heap.Valid → Mutable.Value.HasType heap (.ref address) "list[int]" →
      HeapCalls.Condition program (inputEnv address n) Solvers.P24Ast.requires_nth_lexicographic_permutation_0 heap →
      (HeapCalls.call program "nth_lexicographic_permutation" [.ref address, number n] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (address : Nat) (n : Int), heap.Valid → Mutable.Value.HasType heap (.ref address) "list[int]" →
      HeapCalls.Condition program (inputEnv address n) Solvers.P24Ast.requires_nth_lexicographic_permutation_0 heap →
      (HeapCalls.call P24Ast.module "naive_text" [.ref address, number n] heap).Supported) := by
  constructor
  · intro heap address n _ typed requires
    obtain ⟨digits, found⟩ := Mutable.Value.ints_of_hasType typed
    obtain ⟨_, lower, upper⟩ := condition_domain digits n heap address found requires
    let rank := (n - 1).toNat
    have index : (rank : Int) + 1 = n := by dsimp [rank]; omega
    obtain ⟨_, returned⟩ := optimized.result heap address digits rank found (by omega)
    simpa only [index] using returned.returns.supported
  · intro heap address n valid typed _
    exact data.«project-euler-statements».data.py.project_euler_statements.p24.naive_text.«Supported» heap address n valid typed

end solvers.«24»
