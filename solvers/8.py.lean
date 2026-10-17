import «.formalsw».solvers.«8.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p8.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P8
import Mathlib.Tactic

namespace ProjectEulerSolutions.P8.Python
open data.«project-euler-statements».data.py.project_euler_statements.p8 (windowValues)

def window (digits : List Int) (start width : Nat) : List Int := (digits.drop start).take width
def windowProduct (digits : List Int) (start width : Nat) : Int := (window digits start width).prod

def maximum (values : List Int) : Int := values.foldl max 0
def maximumWindow (digits : List Int) (width : Nat) : Int := maximum (windowValues width digits)

/-- A rolling window cancels its leaving factor exactly, even for negative integers. -/
theorem windowProduct_roll (digits : List Int) (start width : Nat)
    (within : start + (width + 1) < digits.length)
    (nonzero : digits[start]'(by omega) ≠ 0) :
    Int.fdiv (windowProduct digits start (width + 1)) digits[start] * digits[start + (width + 1)] =
      windowProduct digits (start + 1) (width + 1) := by
  have first : start < digits.length := by omega
  have enough : width < (digits.drop (start + 1)).length := by simp; omega
  unfold windowProduct window
  rw [List.drop_eq_getElem_cons first, List.take_succ_cons, List.prod_cons,
    Int.mul_fdiv_cancel_left _ nonzero, List.take_succ_eq_append_getElem enough]
  simp [List.prod_append, List.getElem_drop, Nat.add_comm, Nat.add_left_comm]

theorem windowProduct_step (digits : List Int) (start stop : Nat)
    (before : start < stop) (bounded : stop ≤ digits.length) :
    windowProduct digits start (stop - start) =
      digits[start]'(by omega) * windowProduct digits (start + 1) (stop - (start + 1)) := by
  have width : stop - start = (stop - (start + 1)) + 1 := by omega
  unfold windowProduct window
  rw [List.drop_eq_getElem_cons (by omega : start < digits.length), width,
    List.take_succ_cons, List.prod_cons]

private theorem foldl_max (values : List Int) (a b : Int) :
    values.foldl max (max a b) = max a (values.foldl max b) := by
  induction values generalizing a b with
  | nil => rfl
  | cons head tail ih => simpa only [List.foldl_cons, max_assoc] using ih a (max b head)

theorem maximum_cons (value : Int) (values : List Int) :
    maximum (value :: values) = max value (maximum values) := by
  simp only [maximum, List.foldl_cons, max_comm (0 : Int) value]
  exact foldl_max values value 0

theorem maximum_nonnegative (values : List Int) : 0 ≤ maximum values := by
  induction values with
  | nil => rfl
  | cons head tail ih => rw [maximum_cons]; exact le_trans ih (le_max_right _ _)

theorem maximumWindow_nonnegative (digits : List Int) (width : Nat) :
    0 ≤ maximumWindow digits width := maximum_nonnegative _

theorem maximumWindow_short (digits : List Int) (width : Nat) (short : digits.length < width) :
    maximumWindow digits width = 0 := by
  cases digits with
  | nil => rfl
  | cons head tail =>
      have shorter : (tail.length : Int) + 1 < (width : Int) := by simpa using (show ((head :: tail).length : Int) < width by exact_mod_cast short)
      simp [maximumWindow, windowValues, shorter, maximum]

theorem maximumWindow_cons (head : Int) (tail : List Int) (width : Nat)
    (enough : width ≤ tail.length + 1) :
    maximumWindow (head :: tail) width =
      max ((head :: tail).take width).prod (maximumWindow tail width) := by
  have longer : ¬ (tail.length : Int) + 1 < (width : Int) := by omega
  simp [maximumWindow, windowValues, longer, FormalSW.Py.Builtin.Slice.prefixLength_nonnegative,
    min_eq_left enough, maximum_cons]

/-- A zero cannot improve a nonnegative running maximum. -/
theorem maximumWindow_zero (digits : List Int) (width : Nat) (positive : 0 < width) :
    maximumWindow (0 :: digits) width = maximumWindow digits width := by
  by_cases enough : width ≤ digits.length + 1
  · rw [maximumWindow_cons _ _ _ enough]
    obtain ⟨width, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : width ≠ 0)
    simp only [List.take_succ_cons, List.prod_cons, zero_mul]
    exact max_eq_right (maximum_nonnegative _)
  · rw [maximumWindow_short _ _ (by simp; omega), maximumWindow_short _ _ (by omega)]

/-- Windows crossing a zero contribute nothing to either segment's maximum. -/
theorem maximumWindow_split (left right : List Int) (width : Nat) (positive : 0 < width) :
    maximumWindow (left ++ 0 :: right) width =
      max (maximumWindow left width) (maximumWindow right width) := by
  induction left with
  | nil =>
      rw [List.nil_append, maximumWindow_zero _ _ positive]
      exact (max_eq_right (maximum_nonnegative _)).symm
  | cons head tail ih =>
      by_cases enough : width ≤ tail.length + 1
      · rw [List.cons_append, maximumWindow_cons _ _ _ (by simp; omega),
          maximumWindow_cons _ _ _ enough, ih]
        have take : (head :: (tail ++ 0 :: right)).take width = (head :: tail).take width :=
          List.take_append_of_le_length (l₁ := head :: tail) (l₂ := 0 :: right) (by simpa using enough)
        rw [take, max_assoc]
      · rw [maximumWindow_short (head :: tail) width (by simp; omega),
          max_eq_right (maximumWindow_nonnegative right width)]
        by_cases long : width ≤ (tail ++ 0 :: right).length + 1
        · rw [List.cons_append, maximumWindow_cons _ _ _ long, ih,
            maximumWindow_short tail width (by omega), max_eq_right (maximumWindow_nonnegative right width)]
          have zero : ((head :: (tail ++ 0 :: right)).take width).prod = 0 := by
            change (((head :: tail) ++ 0 :: right).take width).prod = 0
            apply List.prod_eq_zero_iff.mpr
            apply List.mem_of_getElem? (i := (head :: tail).length)
            rw [List.getElem?_take_of_lt (by simp; omega), List.getElem?_append_right (by omega)]
            simp
          rw [zero, max_eq_right (maximumWindow_nonnegative right width)]
        · rw [maximumWindow_short _ _ (by simp_all),
            maximumWindow_short right width (by simp at long; omega)]

/-- Products created by rolling the right edge through a half-open interval. -/
def rollingMaximum (digits : List Int) (width start stop : Nat) : Int :=
  maximum ((List.range' start (stop - start)).map
    (fun index => windowProduct digits (index + 1 - width) width))

theorem rollingMaximum_done (digits : List Int) (width index : Nat) :
    rollingMaximum digits width index index = 0 := by simp [rollingMaximum, maximum]

theorem rollingMaximum_step (digits : List Int) (width start stop : Nat) (before : start < stop) :
    rollingMaximum digits width start stop =
      max (windowProduct digits (start + 1 - width) width) (rollingMaximum digits width (start + 1) stop) := by
  have count : stop - start = (stop - (start + 1)) + 1 := by omega
  simp only [rollingMaximum, count, List.range'_succ, List.map_cons, maximum_cons]

/-- The recursive specification enumerates exactly the valid window starts. -/
theorem windowValues_drop (digits : List Int) (width start : Nat) (positive : 0 < width)
    (bounded : start ≤ digits.length) :
    windowValues width (digits.drop start) =
      (List.range' start (digits.length - start + 1 - width)).map (fun index => windowProduct digits index width) := by
  by_cases enough : width ≤ digits.length - start
  · have valid : start < digits.length := by omega
    have count : digits.length - start + 1 - width =
        (digits.length - (start + 1) + 1 - width) + 1 := by omega
    have longer : ¬ ((digits[start] :: digits.drop (start + 1)).length : Int) < (width : Int) := by
      simp only [List.length_cons, List.length_drop]; omega
    rw [List.drop_eq_getElem_cons valid, windowValues, if_neg longer,
      windowValues_drop digits width (start + 1) positive (by omega), count, List.range'_succ, List.map_cons]
    congr 1
    simp only [FormalSW.Py.Builtin.Slice.prefixLength_nonnegative,
      List.length_cons, List.length_drop, min_eq_left (by omega : width ≤ digits.length - (start + 1) + 1)]
    rw [windowProduct, window, List.drop_eq_getElem_cons valid]
  · have short : (digits.drop start).length < width := by simp; omega
    have empty : windowValues width (digits.drop start) = [] := by
      generalize digits.drop start = xs at short ⊢
      cases xs with
      | nil => rfl
      | cons head tail =>
          have short : (tail.length : Int) + 1 < (width : Int) := by simp at short; omega
          simp [windowValues, short]
    rw [empty]
    simp [show digits.length - start + 1 - width = 0 by omega]
termination_by digits.length - start

theorem maximumWindow_indexed (digits : List Int) (width start : Nat) (positive : 0 < width)
    (bounded : start ≤ digits.length) :
    maximumWindow (digits.drop start) width =
      maximum ((List.range' start (digits.length - start + 1 - width)).map
        (fun index => windowProduct digits index width)) := by
  rw [maximumWindow, windowValues_drop digits width start positive bounded]

/-- The initial product and subsequent rolling products cover every full window. -/
theorem maximumWindow_rolling (digits : List Int) (width : Nat) (positive : 0 < width)
    (enough : width ≤ digits.length) :
    maximumWindow digits width =
      max (windowProduct digits 0 width) (rollingMaximum digits width width digits.length) := by
  have count : digits.length - 0 + 1 - width = (digits.length - width) + 1 := by omega
  have indexed := maximumWindow_indexed digits width 0 positive (by omega)
  rw [List.drop_zero, count, List.range'_succ, List.map_cons, maximum_cons] at indexed
  rw [indexed]
  congr 1
  unfold rollingMaximum
  congr 1
  apply List.ext_getElem
  · simp
  · intro index left right
    simp only [List.getElem_map, List.getElem_range']
    congr 1
    omega

theorem windowProduct_slice (digits : List Int) (start stop index width : Nat)
    (inside : start + index + width ≤ stop) :
    windowProduct ((digits.drop start).take (stop - start)) index width =
      windowProduct digits (start + index) width := by
  simp only [windowProduct, window, List.drop_take, List.drop_drop, List.take_take,
    min_eq_left (by omega : width ≤ stop - start - index)]

theorem maximumWindow_segment (digits : List Int) (start stop width : Nat)
    (positive : 0 < width) (enough : start + width ≤ stop) (bounded : stop ≤ digits.length) :
    maximumWindow ((digits.drop start).take (stop - start)) width =
      max (windowProduct digits start width) (rollingMaximum digits width (start + width) stop) := by
  let segment := (digits.drop start).take (stop - start)
  have length : segment.length = stop - start := by
    simp [segment, min_eq_left (by omega : stop - start ≤ digits.length - start)]
  rw [maximumWindow_rolling segment width positive (by rw [length]; omega),
    windowProduct_slice digits start stop 0 width (by omega), Nat.add_zero]
  congr 1
  unfold rollingMaximum
  congr 1
  apply List.ext_getElem
  · simp only [List.length_map, List.length_range', length]; omega
  · intro index left right
    have within : index < stop - start - width := by simpa only [List.length_map, List.length_range', length] using left
    simp only [List.getElem_map, List.getElem_range']
    rw [windowProduct_slice digits start stop (width + 1 * index + 1 - width) width (by omega)]
    congr 1
    omega

/-- Skipping leading zeroes does not change any positive-width maximum. -/
theorem maximumWindow_skip (digits : List Int) (width start stop : Nat)
    (positive : 0 < width) (ordered : start ≤ stop) (bounded : stop ≤ digits.length)
    (zeroes : ∀ index, start ≤ index → (bound : index < stop) → digits[index]'(by omega) = 0) :
    maximumWindow (digits.drop start) width = maximumWindow (digits.drop stop) width := by
  by_cases equal : start = stop
  · rw [equal]
  · have valid : start < digits.length := by omega
    rw [List.drop_eq_getElem_cons valid, zeroes start (by omega) (by omega), maximumWindow_zero _ _ positive]
    exact maximumWindow_skip digits width (start + 1) stop positive (by omega) bounded
      (fun index lower upper => zeroes index (by omega) upper)
termination_by stop - start

/-- A zero at the segment boundary separates its windows from all later windows. -/
theorem maximumWindow_boundary (digits : List Int) (width start stop : Nat)
    (positive : 0 < width) (ordered : start ≤ stop) (bounded : stop ≤ digits.length)
    (zero : ∀ bound : stop < digits.length, digits[stop]'bound = 0) :
    maximumWindow (digits.drop start) width =
      max (maximumWindow ((digits.drop start).take (stop - start)) width)
        (maximumWindow (digits.drop stop) width) := by
  have split : (digits.drop start).take (stop - start) ++ digits.drop stop = digits.drop start := by
    have h := List.take_append_drop (stop - start) (digits.drop start)
    simpa only [List.drop_drop, Nat.add_sub_of_le ordered] using h
  by_cases valid : stop < digits.length
  · have tail : digits.drop stop = 0 :: digits.drop (stop + 1) := by
      rw [List.drop_eq_getElem_cons valid, zero valid]
    conv_lhs => rw [← split]
    rw [tail, maximumWindow_split _ _ _ positive, maximumWindow_zero _ _ positive]
  · have empty : digits.drop stop = [] := List.drop_eq_nil_iff.mpr (by omega)
    rw [empty, List.append_nil] at split
    rw [split, empty]
    exact (max_eq_left (maximumWindow_nonnegative _ _)).symm

theorem maximumWindow_zeroWidth (digits : List Int) :
    maximumWindow digits 0 = if digits.isEmpty then 0 else 1 := by
  induction digits with
  | nil => rfl
  | cons head tail ih =>
      rw [maximumWindow_cons head tail 0 (by omega), ih]
      cases tail.isEmpty <;> simp

end ProjectEulerSolutions.P8.Python

namespace solvers.«8»
open FormalSW.PyAst
open data.«project-euler-statements».data.py.project_euler_statements.p8 (windowValues)
private abbrev program := Solvers.P8Ast.module
private def ints (xs : List Int) : List Mutable.Value := xs.map fun x => .scalar (.int x)

private def scanTest (name : String) (zeros : Bool) : Expr :=
  .boolOp .and [
    .compare (.name name) [(.lt, .name "digit_count")],
    .compare (.subscript (.name "digits") (.name name))
      [(if zeros then .eq else .notEq, .constant (.int 0))]]

private def scanLoop (name : String) (zeros : Bool) : Stmt :=
  .whileDo (scanTest name zeros) [.augAssign (.name name) .add (.constant (.int 1))] []

private theorem read_digit (digits : List Int) (heap : Mutable.Heap) (address position : Nat)
    (env : HeapCalls.Env) (indexName : String)
    (found : heap[address]? = some (.list (ints digits))) (valid : position < digits.length)
    (receiver : env "digits" = some (.ref address))
    (index : env indexName = some (.scalar (.int position))) :
    HeapCalls.Evaluates program env (.subscript (.name "digits") (.name indexName))
      heap (.scalar (.int digits[position])) heap := by
  apply HeapCalls.Evaluates.subscript (by intros a b c h; cases h)
    (HeapCalls.Evaluates.lookup receiver) (HeapCalls.Evaluates.lookup index)
  simpa [ints] using Mutable.getItem_list heap address position (ints digits) found (by simpa [ints] using valid)

/-- The bound check short-circuits before attempting an out-of-range read. -/
private theorem scanTest_end (digits : List Int) (heap : Mutable.Heap) (position : Nat)
    (env : HeapCalls.Env) (name : String) (zeros : Bool)
    (index : env name = some (.scalar (.int position)))
    (count : env "digit_count" = some (.scalar (.int digits.length)))
    (finished : digits.length ≤ position) :
    HeapCalls.Evaluates program env (scanTest name zeros) heap (.scalar (.bool false)) heap := by
  apply HeapCalls.Evaluates.boolean_stop (flag := false)
  · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup index)
      (HeapCalls.Evaluates.lookup count) (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
  · rfl
  · rfl

private theorem scanTest_read (digits : List Int) (heap : Mutable.Heap) (address position : Nat)
    (env : HeapCalls.Env) (name : String) (zeros : Bool)
    (found : heap[address]? = some (.list (ints digits))) (valid : position < digits.length)
    (receiver : env "digits" = some (.ref address))
    (index : env name = some (.scalar (.int position)))
    (count : env "digit_count" = some (.scalar (.int digits.length))) :
    HeapCalls.Evaluates program env (scanTest name zeros) heap
      (.scalar (.bool (decide (digits[position] = 0) == zeros))) heap := by
  apply HeapCalls.Evaluates.boolean_next (value := .scalar (.bool true)) (middle := heap) (flag := true)
  · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup index)
      (HeapCalls.Evaluates.lookup count) (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, valid])
  · rfl
  · rfl
  · apply HeapCalls.Evaluates.boolean_last (flag := decide (digits[position] = 0) == zeros)
    · apply HeapCalls.Evaluates.compare (read_digit digits heap address position env name found valid receiver index)
        HeapCalls.Evaluates.constant
      cases zeros <;> simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl

private theorem setLocal_shadow (env : HeapCalls.Env) (name : String) (a b : Mutable.Value) :
    HeapCalls.setLocal (HeapCalls.setLocal env name a) name b = HeapCalls.setLocal env name b := by
  funext query
  by_cases equal : query = name <;> simp [HeapCalls.setLocal, equal]

private theorem setLocal_current (env : HeapCalls.Env) (name : String) (value : Mutable.Value)
    (current : env name = some value) : HeapCalls.setLocal env name value = env := by
  funext query
  by_cases equal : query = name <;> simp_all [HeapCalls.setLocal]

private theorem scan_bounds (digits : List Int) (start : Nat) (zeros : Bool) (bounded : start ≤ digits.length) :
    ∃ stop, start ≤ stop ∧ ∃ upper : stop ≤ digits.length,
      (∀ index, start ≤ index → (bound : index < stop) →
        (decide (digits[index]'(by omega) = 0) == zeros) = true) ∧
      (∀ bound : stop < digits.length, (decide (digits[stop]'bound = 0) == zeros) = false) := by
  by_cases valid : start < digits.length
  · by_cases selected : (decide (digits[start] = 0) == zeros) = true
    · obtain ⟨stop, ordered, bounded, inside, outside⟩ := scan_bounds digits (start + 1) zeros (by omega)
      refine ⟨stop, by omega, bounded, ?_, outside⟩
      intro index lower upper
      by_cases equal : index = start
      · subst index; exact selected
      · exact inside index (by omega) upper
    · exact ⟨start, le_rfl, bounded, by intros; omega, by intros; simpa using selected⟩
  · exact ⟨start, le_rfl, bounded, by intros; omega, by intros; omega⟩
termination_by digits.length - start

/-- Both source scans traverse exactly their selected interval and preserve the input heap. -/
private theorem scan_result (digits : List Int) (heap : Mutable.Heap) (address start stop : Nat)
    (env : HeapCalls.Env) (name : String) (zeros : Bool)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (count : env "digit_count" = some (.scalar (.int digits.length)))
    (notDigits : "digits" ≠ name) (notCount : "digit_count" ≠ name)
    (ordered : start ≤ stop) (bounded : stop ≤ digits.length)
    (selected : ∀ index, start ≤ index → (bound : index < stop) →
      (decide (digits[index]'(by omega) = 0) == zeros) = true)
    (finished : ∀ bound : stop < digits.length,
      (decide (digits[stop]'bound = 0) == zeros) = false) :
    HeapCalls.BlockContinues program (HeapCalls.setLocal env name (.scalar (.int start)))
      [scanLoop name zeros] heap (HeapCalls.setLocal env name (.scalar (.int stop))) heap := by
  by_cases atEnd : start = stop
  · subst start
    apply HeapCalls.BlockContinues.while_loop (value := .scalar (.bool false)) (flag := false) (middle := heap)
    · by_cases valid : stop < digits.length
      · simpa only [finished valid] using scanTest_read digits heap address stop
          (HeapCalls.setLocal env name (.scalar (.int stop))) name zeros found valid
          (by simp [HeapCalls.setLocal, notDigits, receiver])
          (by simp [HeapCalls.setLocal]) (by simp [HeapCalls.setLocal, notCount, count])
      · exact scanTest_end digits heap stop _ name zeros
          (by simp [HeapCalls.setLocal]) (by simp [HeapCalls.setLocal, notCount, count]) (by omega)
    · rfl
    · exact HeapCalls.BlockContinues.nil
  · have before : start < stop := by omega
    apply HeapCalls.BlockContinues.while_loop (value := .scalar (.bool true)) (flag := true) (middle := heap)
    · simpa only [selected start (by omega) before] using scanTest_read digits heap address start
        (HeapCalls.setLocal env name (.scalar (.int start))) name zeros found (by omega)
        (by simp [HeapCalls.setLocal, notDigits, receiver])
        (by simp [HeapCalls.setLocal]) (by simp [HeapCalls.setLocal, notCount, count])
    · rfl
    · apply HeapCalls.BlockContinues.append (middleEnv := HeapCalls.setLocal env name (.scalar (.int (start + 1 : Nat))))
        (middleHeap := heap)
      · have increment := HeapCalls.BlockContinues.augAssign (module := program)
          (env := HeapCalls.setLocal env name (.scalar (.int start))) (name := name)
          (heap := heap) (middle := heap) (finalHeap := heap) (op := .add)
          (left := .scalar (.int start)) (right := .scalar (.int 1))
          (value := .scalar (.int (start + 1 : Nat)))
          (by simp [HeapCalls.setLocal]) HeapCalls.Evaluates.constant
          (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
        simpa only [setLocal_shadow] using increment
      · exact scan_result digits heap address (start + 1) stop env name zeros found receiver count
          notDigits notCount (by omega) bounded (fun index lower upper => selected index (by omega) upper) finished
termination_by stop - start

private def productBody : List Stmt :=
  [.augAssign (.name "prod") .mult (.subscript (.name "digits") (.name "t"))]

/-- The first product loop multiplies each indexed digit once, without allocating or mutating. -/
private theorem product_range (digits : List Int) (heap : Mutable.Heap) (address start stop : Nat)
    (env : HeapCalls.Env) (product : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (initial : env "prod" = some (.scalar (.int product)))
    (ordered : start ≤ stop) (bounded : stop ≤ digits.length) :
    ∃ finalEnv,
      HeapCalls.BlockContinues program env [.rangeStepLoop "t" start stop 1 productBody []]
        heap finalEnv heap ∧
      finalEnv "prod" = some (.scalar (.int
        (product * ProjectEulerSolutions.P8.Python.windowProduct digits start (stop - start)))) ∧
      ∀ name, name ≠ "prod" → name ≠ "t" → finalEnv name = env name := by
  by_cases atEnd : start = stop
  · subst start
    refine ⟨env, HeapCalls.BlockContinues.range_done ?_ HeapCalls.BlockContinues.nil, ?_, ?_⟩
    · simp [Calls.rangeContinues]
    · simpa [ProjectEulerSolutions.P8.Python.windowProduct, ProjectEulerSolutions.P8.Python.window] using initial
    · intros; rfl
  · have before : start < stop := by omega
    have valid : start < digits.length := by omega
    let nextEnv := HeapCalls.setLocal (HeapCalls.setLocal env "t" (.scalar (.int start)))
      "prod" (.scalar (.int (product * digits[start])))
    obtain ⟨finalEnv, rest, result, unchanged⟩ :=
      product_range digits heap address (start + 1) stop nextEnv (product * digits[start]) found
        (by simp [nextEnv, HeapCalls.setLocal, receiver])
        (by simp [nextEnv, HeapCalls.setLocal]) (by omega) bounded
    refine ⟨finalEnv, ?_, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues, before])
      apply HeapCalls.BlockContinues.append (middleEnv := nextEnv) (middleHeap := heap)
      · apply HeapCalls.BlockContinues.augAssign
          (left := .scalar (.int product)) (right := .scalar (.int digits[start])) (middle := heap)
        · simpa [HeapCalls.setLocal] using initial
        · exact read_digit digits heap address start _ "t" found valid
            (by simp [HeapCalls.setLocal, receiver]) (by simp [HeapCalls.setLocal])
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
      · simpa [Calls.loopContinuation, Calls.loopControl, productBody] using rest
    · rw [ProjectEulerSolutions.P8.Python.windowProduct_step digits start stop before bounded, ← mul_assoc]
      exact result
    · intro name notProduct notIndex
      rw [unchanged name notProduct notIndex]
      simp [nextEnv, HeapCalls.setLocal, notProduct, notIndex]
termination_by stop - start

/-- Python floor division agrees with Lean's floor quotient for either divisor sign. -/
private theorem divide (a b : Int) (heap : Mutable.Heap) (nonzero : b ≠ 0) :
    Mutable.binary .floorDiv (.scalar (.int a)) (.scalar (.int b)) heap =
      some (.ok (.scalar (.int (a.fdiv b))), heap) := by
  by_cases positive : 0 < b
  · simp [Mutable.binary, scalarBinary, nonzero, positive,
      Int.fdiv_eq_ediv_of_nonneg a (by omega : 0 ≤ b)]
  · have quotient : a.fdiv b = (-a) / (-b) := by
      rw [← Int.neg_fdiv_neg, Int.fdiv_eq_ediv_of_nonneg _ (by omega : 0 ≤ -b)]
    simp [Mutable.binary, scalarBinary, nonzero, positive, quotient]

private def rollExpression : Expr :=
  .binOp (.binOp (.name "prod") .floorDiv
    (.subscript (.name "digits") (.binOp (.name "t") .sub (.name "k")))) .mult
    (.subscript (.name "digits") (.name "t"))

private theorem roll_value (digits : List Int) (heap : Mutable.Heap) (address index width : Nat)
    (env : HeapCalls.Env) (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (counter : env "t" = some (.scalar (.int index)))
    (size : env "k" = some (.scalar (.int width)))
    (product : env "prod" = some (.scalar (.int
      (ProjectEulerSolutions.P8.Python.windowProduct digits (index - width) width))))
    (positive : 0 < width) (ordered : width ≤ index) (bounded : index < digits.length)
    (nonzero : digits[index - width]'(by omega) ≠ 0) :
    HeapCalls.Evaluates program env rollExpression heap
      (.scalar (.int (ProjectEulerSolutions.P8.Python.windowProduct digits (index + 1 - width) width))) heap := by
  let old := ProjectEulerSolutions.P8.Python.windowProduct digits (index - width) width
  have roll : (old.fdiv digits[index - width]) * digits[index] =
      ProjectEulerSolutions.P8.Python.windowProduct digits (index + 1 - width) width := by
    have h := ProjectEulerSolutions.P8.Python.windowProduct_roll digits (index - width) (width - 1)
      (by omega) nonzero
    simpa only [show width - 1 + 1 = width by omega,
      show index - width + width = index by omega,
      show index - width + 1 = index + 1 - width by omega] using h
  apply HeapCalls.Evaluates.binary (av := .scalar (.int (old.fdiv digits[index - width])))
    (bv := .scalar (.int digits[index])) (middle := heap) (afterRight := heap)
  · apply HeapCalls.Evaluates.binary (av := .scalar (.int old))
      (bv := .scalar (.int digits[index - width])) (middle := heap) (afterRight := heap)
    · exact HeapCalls.Evaluates.lookup product
    · apply HeapCalls.Evaluates.subscript (by intros a b c h; cases h)
        (HeapCalls.Evaluates.lookup receiver) (position := .scalar (.int (index - width : Nat)))
        (afterIndex := heap)
      · exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup counter) (HeapCalls.Evaluates.lookup size)
          (by simp [Mutable.binary, scalarBinary, Int.natCast_sub ordered])
      · simpa [ints] using Mutable.getItem_list heap address (index - width) (ints digits) found (by simp [ints]; omega)
    · exact divide old digits[index - width] heap nonzero
  · exact read_digit digits heap address index env "t" found bounded receiver counter
  · simpa [Mutable.binary, scalarBinary] using roll

private def updateBest : Stmt :=
  .ifThen (.compare (.name "prod") [(.gt, .name "best")]) [.assign [.name "best"] (.name "prod")] []

private theorem update_best (env : HeapCalls.Env) (heap : Mutable.Heap) (product best : Int)
    (current : env "prod" = some (.scalar (.int product)))
    (previous : env "best" = some (.scalar (.int best))) :
    ∃ finalEnv, HeapCalls.BlockContinues program env [updateBest] heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int (max best product))) ∧
      ∀ name, name ≠ "best" → finalEnv name = env name := by
  by_cases larger : best < product
  · refine ⟨HeapCalls.setLocal env "best" (.scalar (.int product)), ?_, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.branch (value := .scalar (.bool true)) (flag := true) (middle := heap)
      · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup current) (HeapCalls.Evaluates.lookup previous)
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, larger])
      · rfl
      · exact HeapCalls.BlockContinues.assign (HeapCalls.Evaluates.lookup current)
    · simp [HeapCalls.setLocal, max_eq_right (le_of_lt larger)]
    · intro name different; simp [HeapCalls.setLocal, different]
  · refine ⟨env, ?_, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.branch (value := .scalar (.bool false)) (flag := false) (middle := heap)
      · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup current) (HeapCalls.Evaluates.lookup previous)
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, larger])
      · rfl
      · exact HeapCalls.BlockContinues.nil
    · simpa [max_eq_left (by omega : product ≤ best)] using previous
    · intros; rfl

open ProjectEulerSolutions.P8.Python

private def rollingBody : List Stmt := [.assign [.name "prod"] rollExpression, updateBest]

/-- Each rolling iteration divides by a nonzero leaving digit and incorporates the new product. -/
private theorem rolling_range (digits : List Int) (heap : Mutable.Heap) (address width start stop : Nat)
    (env : HeapCalls.Env) (best : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (size : env "k" = some (.scalar (.int width)))
    (product : env "prod" = some (.scalar (.int (windowProduct digits (start - width) width))))
    (previous : env "best" = some (.scalar (.int best)))
    (nonnegative : 0 ≤ best) (positive : 0 < width) (ordered : width ≤ start)
    (interval : start ≤ stop) (bounded : stop ≤ digits.length)
    (nonzero : ∀ index, start ≤ index → (bound : index < stop) → digits[index - width]'(by omega) ≠ 0) :
    ∃ finalEnv,
      HeapCalls.BlockContinues program env [.rangeStepLoop "t" start stop 1 rollingBody []]
        heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int (max best (rollingMaximum digits width start stop)))) ∧
      ∀ name, name ≠ "prod" → name ≠ "t" → name ≠ "best" → finalEnv name = env name := by
  by_cases atEnd : start = stop
  · subst start
    refine ⟨env, HeapCalls.BlockContinues.range_done ?_ HeapCalls.BlockContinues.nil, ?_, ?_⟩
    · simp [Calls.rangeContinues]
    · simpa [rollingMaximum_done, max_eq_left nonnegative] using previous
    · intros; rfl
  · have before : start < stop := by omega
    have valid : start < digits.length := by omega
    let nextProduct := windowProduct digits (start + 1 - width) width
    let indexEnv := HeapCalls.setLocal env "t" (.scalar (.int start))
    let productEnv := HeapCalls.setLocal indexEnv "prod" (.scalar (.int nextProduct))
    have first : HeapCalls.BlockContinues program indexEnv [.assign [.name "prod"] rollExpression]
        heap productEnv heap :=
      HeapCalls.BlockContinues.assign (roll_value digits heap address start width indexEnv found
        (by simp [indexEnv, HeapCalls.setLocal, receiver]) (by simp [indexEnv, HeapCalls.setLocal])
        (by simp [indexEnv, HeapCalls.setLocal, size]) (by simp [indexEnv, HeapCalls.setLocal, product])
        positive ordered valid (nonzero start (by omega) before))
    obtain ⟨bestEnv, update, current, preserved⟩ := update_best productEnv heap nextProduct best
      (by simp [productEnv, HeapCalls.setLocal]) (by simp [productEnv, indexEnv, HeapCalls.setLocal, previous])
    obtain ⟨finalEnv, rest, result, unchanged⟩ :=
      rolling_range digits heap address width (start + 1) stop bestEnv (max best nextProduct) found
        (by rw [preserved "digits" (by decide)]; simp [productEnv, indexEnv, HeapCalls.setLocal, receiver])
        (by rw [preserved "k" (by decide)]; simp [productEnv, indexEnv, HeapCalls.setLocal, size])
        (by rw [preserved "prod" (by decide)]; simp [productEnv, HeapCalls.setLocal, nextProduct]) current
        (le_trans nonnegative (le_max_left _ _)) positive (by omega) (by omega) bounded
        (fun index lower upper => nonzero index (by omega) upper)
    refine ⟨finalEnv, ?_, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues, before])
      apply HeapCalls.BlockContinues.append first
      apply HeapCalls.BlockContinues.append update
      simpa [Calls.loopContinuation, Calls.loopControl, rollingBody, updateBest] using rest
    · rw [rollingMaximum_step digits width start stop before, ← max_assoc]
      exact result
    · intro name notProduct notIndex notBest
      rw [unchanged name notProduct notIndex notBest, preserved name notBest]
      simp [productEnv, indexEnv, HeapCalls.setLocal, notProduct, notIndex]
termination_by stop - start

/-- The source statements executed for a segment long enough to contain a window. -/
private def segmentBody : List Stmt := [
  .assign [.name "prod"] (.constant (.int 1)),
  .forIn (.name "t") (.call (.name "range") [.name "i", .binOp (.name "i") .add (.name "k")]) productBody [],
  updateBest,
  .forIn (.name "t") (.call (.name "range") [.binOp (.name "i") .add (.name "k"), .name "j"]) rollingBody []]

private theorem segment_result (digits : List Int) (heap : Mutable.Heap) (address start stop width : Nat)
    (env : HeapCalls.Env) (best : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (size : env "k" = some (.scalar (.int width)))
    (left : env "i" = some (.scalar (.int start)))
    (right : env "j" = some (.scalar (.int stop)))
    (previous : env "best" = some (.scalar (.int best)))
    (nonnegative : 0 ≤ best) (positive : 0 < width)
    (enough : start + width ≤ stop) (bounded : stop ≤ digits.length)
    (nonzero : ∀ index, start ≤ index → (bound : index < stop) → digits[index]'(by omega) ≠ 0) :
    ∃ finalEnv, HeapCalls.BlockContinues program env segmentBody heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int
        (max best (maximumWindow ((digits.drop start).take (stop - start)) width)))) ∧
      ∀ name, name ≠ "prod" → name ≠ "t" → name ≠ "best" → finalEnv name = env name := by
  let initialEnv := HeapCalls.setLocal env "prod" (.scalar (.int 1))
  obtain ⟨productEnv, multiply, product, preservedProduct⟩ :=
    product_range digits heap address start (start + width) initialEnv 1 found
      (by simp [initialEnv, HeapCalls.setLocal, receiver]) (by simp [initialEnv, HeapCalls.setLocal])
      (by omega) (by omega)
  have product : productEnv "prod" = some (.scalar (.int (windowProduct digits start width))) := by
    simpa using product
  obtain ⟨bestEnv, update, current, preservedBest⟩ := update_best productEnv heap (windowProduct digits start width) best
    product (by rw [preservedProduct "best" (by decide) (by decide)]; simpa [initialEnv, HeapCalls.setLocal] using previous)
  have kept : ∀ name, name ≠ "prod" → name ≠ "t" → name ≠ "best" → bestEnv name = env name := by
    intro name notProduct notIndex notBest
    rw [preservedBest name notBest, preservedProduct name notProduct notIndex]
    simp [initialEnv, HeapCalls.setLocal, notProduct]
  obtain ⟨finalEnv, roll, result, preservedRoll⟩ :=
    rolling_range digits heap address width (start + width) stop bestEnv (max best (windowProduct digits start width)) found
      (by rw [kept "digits" (by decide) (by decide) (by decide)]; exact receiver)
      (by rw [kept "k" (by decide) (by decide) (by decide)]; exact size)
      (by rw [preservedBest "prod" (by decide)]; simpa using product) current
      (le_trans nonnegative (le_max_left _ _)) positive (by omega) enough bounded
      (fun index lower upper => nonzero (index - width) (by omega) (by omega))
  refine ⟨finalEnv, ?_, ?_, ?_⟩
  · apply HeapCalls.BlockContinues.append (body := [.assign [.name "prod"] (.constant (.int 1))])
      (HeapCalls.BlockContinues.assign HeapCalls.Evaluates.constant)
    apply HeapCalls.BlockContinues.append (body := [.forIn (.name "t")
      (.call (.name "range") [.name "i", .binOp (.name "i") .add (.name "k")]) productBody []])
    · apply HeapCalls.BlockContinues.for_range (values := [.scalar (.int start), .scalar (.int (start + width : Nat))])
        (middle := heap)
      · apply HeapCalls.Evaluates.Arguments.cons
          (HeapCalls.Evaluates.lookup (by simpa [initialEnv, HeapCalls.setLocal] using left))
        apply HeapCalls.Evaluates.Arguments.cons _ .nil
        exact HeapCalls.Evaluates.binary
          (HeapCalls.Evaluates.lookup (by simpa [initialEnv, HeapCalls.setLocal] using left))
          (HeapCalls.Evaluates.lookup (by simpa [initialEnv, HeapCalls.setLocal] using size))
          (by simp [Mutable.binary, scalarBinary])
      · rfl
      · exact multiply
    · apply HeapCalls.BlockContinues.append update
      apply HeapCalls.BlockContinues.for_range (values := [.scalar (.int (start + width : Nat)), .scalar (.int stop)])
        (middle := heap)
      · apply HeapCalls.Evaluates.Arguments.cons _
          (.cons (HeapCalls.Evaluates.lookup (by rw [kept "j" (by decide) (by decide) (by decide)]; exact right)) .nil)
        exact HeapCalls.Evaluates.binary
          (HeapCalls.Evaluates.lookup (by rw [kept "i" (by decide) (by decide) (by decide)]; exact left))
          (HeapCalls.Evaluates.lookup (by rw [kept "k" (by decide) (by decide) (by decide)]; exact size))
          (by simp [Mutable.binary, scalarBinary])
      · rfl
      · exact roll
  · rw [maximumWindow_segment digits start stop width positive enough bounded, ← max_assoc]
    exact result
  · intro name notProduct notIndex notBest
    rw [preservedRoll name notProduct notIndex notBest, kept name notProduct notIndex notBest]

private def processSegment : List Stmt := [
  .assign [.name "seg_len"] (.binOp (.name "j") .sub (.name "i")),
  .ifThen (.compare (.name "seg_len") [(.gtE, .name "k")]) segmentBody []]

private theorem process_segment (digits : List Int) (heap : Mutable.Heap) (address start stop width : Nat)
    (env : HeapCalls.Env) (best : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (size : env "k" = some (.scalar (.int width)))
    (left : env "i" = some (.scalar (.int start)))
    (right : env "j" = some (.scalar (.int stop)))
    (previous : env "best" = some (.scalar (.int best)))
    (nonnegative : 0 ≤ best) (positive : 0 < width)
    (ordered : start ≤ stop) (bounded : stop ≤ digits.length)
    (nonzero : ∀ index, start ≤ index → (bound : index < stop) → digits[index]'(by omega) ≠ 0) :
    ∃ finalEnv, HeapCalls.BlockContinues program env processSegment heap finalEnv heap ∧
      finalEnv "best" = some (.scalar (.int
        (max best (maximumWindow ((digits.drop start).take (stop - start)) width)))) ∧
      ∀ name, name ≠ "prod" → name ≠ "t" → name ≠ "best" → name ≠ "seg_len" → finalEnv name = env name := by
  let lengthEnv := HeapCalls.setLocal env "seg_len" (.scalar (.int (stop - start : Nat)))
  have lengthStep : HeapCalls.BlockContinues program env
      [.assign [.name "seg_len"] (.binOp (.name "j") .sub (.name "i"))] heap lengthEnv heap :=
    HeapCalls.BlockContinues.assign (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup right)
      (HeapCalls.Evaluates.lookup left) (by simp [Mutable.binary, scalarBinary, Int.natCast_sub ordered]))
  by_cases enough : start + width ≤ stop
  · obtain ⟨finalEnv, execution, result, preserved⟩ := segment_result digits heap address start stop width lengthEnv best found
      (by simp [lengthEnv, HeapCalls.setLocal, receiver]) (by simp [lengthEnv, HeapCalls.setLocal, size])
      (by simp [lengthEnv, HeapCalls.setLocal, left]) (by simp [lengthEnv, HeapCalls.setLocal, right])
      (by simp [lengthEnv, HeapCalls.setLocal, previous]) nonnegative positive enough bounded nonzero
    refine ⟨finalEnv, HeapCalls.BlockContinues.append lengthStep ?_, result, ?_⟩
    · apply HeapCalls.BlockContinues.branch (value := .scalar (.bool true)) (flag := true) (middle := heap)
      · exact HeapCalls.Evaluates.compare (av := .scalar (.int (stop - start : Nat))) (bv := .scalar (.int width))
          (HeapCalls.Evaluates.lookup (by simp [lengthEnv, HeapCalls.setLocal]))
          (HeapCalls.Evaluates.lookup (by simp [lengthEnv, HeapCalls.setLocal, size]))
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
      · rfl
      · exact execution
    · intro name notProduct notIndex notBest notLength
      rw [preserved name notProduct notIndex notBest]
      simp [lengthEnv, HeapCalls.setLocal, notLength]
  · refine ⟨lengthEnv, HeapCalls.BlockContinues.append lengthStep ?_, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.branch (value := .scalar (.bool false)) (flag := false) (middle := heap)
      · exact HeapCalls.Evaluates.compare (av := .scalar (.int (stop - start : Nat))) (bv := .scalar (.int width))
          (HeapCalls.Evaluates.lookup (by simp [lengthEnv, HeapCalls.setLocal]))
          (HeapCalls.Evaluates.lookup (by simp [lengthEnv, HeapCalls.setLocal, size]))
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
      · rfl
      · exact HeapCalls.BlockContinues.nil
    · rw [maximumWindow_short _ _ (by simp; omega), max_eq_left nonnegative]
      simpa [lengthEnv, HeapCalls.setLocal] using previous
    · intro name _ _ _ notLength
      simp [lengthEnv, HeapCalls.setLocal, notLength]

private def outerBody : List Stmt := [
  scanLoop "i" true,
  .ifThen (.compare (.name "i") [(.gtE, .name "digit_count")]) [.breakLoop] [],
  .assign [.name "j"] (.name "i"),
  scanLoop "j" false] ++ processSegment ++ [.assign [.name "i"] (.name "j")]

private def outerLoop : Stmt :=
  .whileDo (.compare (.name "i") [(.lt, .name "digit_count")]) outerBody []

/-- The outer source loop partitions the input at zeroes and preserves every heap object. -/
private theorem outer_result (digits : List Int) (heap : Mutable.Heap) (address width start : Nat)
    (env : HeapCalls.Env) (best : Int)
    (found : heap[address]? = some (.list (ints digits)))
    (receiver : env "digits" = some (.ref address))
    (size : env "k" = some (.scalar (.int width)))
    (count : env "digit_count" = some (.scalar (.int digits.length)))
    (index : env "i" = some (.scalar (.int start)))
    (previous : env "best" = some (.scalar (.int best)))
    (nonnegative : 0 ≤ best) (positive : 0 < width) (bounded : start ≤ digits.length) :
    HeapCalls.BlockReturns program env [outerLoop, .return (.name "best")] heap
      (.scalar (.int (max best (maximumWindow (digits.drop start) width)))) heap := by
  by_cases valid : start < digits.length
  · obtain ⟨first, ordered, firstBound, zeroes, firstNonzero⟩ := scan_bounds digits start true bounded
    have skipped := scan_result digits heap address start first env "i" true found receiver count
      (by decide) (by decide) ordered firstBound zeroes firstNonzero
    rw [setLocal_current env "i" _ index] at skipped
    let zeroEnv := HeapCalls.setLocal env "i" (.scalar (.int first))
    have skip : maximumWindow (digits.drop start) width = maximumWindow (digits.drop first) width :=
      maximumWindow_skip digits width start first positive ordered firstBound
        (fun index lower upper => by simpa using zeroes index lower upper)
    rw [skip]
    apply HeapCalls.BlockReturns.while_loop (value := .scalar (.bool true)) (flag := true) (middle := heap)
    · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup index) (HeapCalls.Evaluates.lookup count)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, valid])
    · rfl
    · apply HeapCalls.BlockContinues.returns skipped
      by_cases exhausted : first = digits.length
      · apply HeapCalls.BlockReturns.branch (value := .scalar (.bool true)) (flag := true) (middle := heap)
        · exact HeapCalls.Evaluates.compare (av := .scalar (.int first)) (bv := .scalar (.int digits.length))
            (HeapCalls.Evaluates.lookup (by simp [HeapCalls.setLocal]))
            (HeapCalls.Evaluates.lookup (by simp [HeapCalls.setLocal, count]))
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, exhausted])
        · rfl
        · apply HeapCalls.BlockReturns.break_loop rfl
          simpa [zeroEnv, exhausted, maximumWindow, maximum, windowValues, max_eq_left nonnegative] using
            HeapCalls.BlockReturns.return_value (module := program) (heap := heap) (rest := [])
              (HeapCalls.Evaluates.lookup (show zeroEnv "best" = some (.scalar (.int best)) by
                simp [zeroEnv, HeapCalls.setLocal, previous]))
      · have firstValid : first < digits.length := by omega
        have firstNotZero : digits[first] ≠ 0 := by simpa using firstNonzero firstValid
        obtain ⟨stop, stopOrdered, stopBound, selected, boundary⟩ := scan_bounds digits first false firstBound
        have progress : first < stop := by
          by_contra failed
          have equal : stop = first := by omega
          subst stop
          have zero : digits[first] = 0 := by simpa using boundary firstValid
          exact firstNotZero zero
        let segmentEnv := HeapCalls.setLocal zeroEnv "j" (.scalar (.int stop))
        have scanned := scan_result digits heap address first stop zeroEnv "j" false found
          (by simp [zeroEnv, HeapCalls.setLocal, receiver]) (by simp [zeroEnv, HeapCalls.setLocal, count])
          (by decide) (by decide) stopOrdered stopBound selected boundary
        have split := maximumWindow_boundary digits width first stop positive stopOrdered stopBound
          (fun bound => by simpa using boundary bound)
        obtain ⟨finalEnv, process, result, preserved⟩ := process_segment digits heap address first stop width segmentEnv best found
          (by simp [segmentEnv, zeroEnv, HeapCalls.setLocal, receiver])
          (by simp [segmentEnv, zeroEnv, HeapCalls.setLocal, size])
          (by simp [segmentEnv, zeroEnv, HeapCalls.setLocal]) (by simp [segmentEnv, HeapCalls.setLocal])
          (by simp [segmentEnv, zeroEnv, HeapCalls.setLocal, previous]) nonnegative positive stopOrdered stopBound
          (fun index lower upper => by simpa using selected index lower upper)
        let nextEnv := HeapCalls.setLocal finalEnv "i" (.scalar (.int stop))
        have next := outer_result digits heap address width stop nextEnv
          (max best (maximumWindow ((digits.drop first).take (stop - first)) width)) found
          (by simp only [nextEnv, HeapCalls.setLocal]
              rw [preserved "digits" (by decide) (by decide) (by decide) (by decide)]
              simp [segmentEnv, zeroEnv, HeapCalls.setLocal, receiver])
          (by simp only [nextEnv, HeapCalls.setLocal]
              rw [preserved "k" (by decide) (by decide) (by decide) (by decide)]
              simp [segmentEnv, zeroEnv, HeapCalls.setLocal, size])
          (by simp only [nextEnv, HeapCalls.setLocal]
              rw [preserved "digit_count" (by decide) (by decide) (by decide) (by decide)]
              simp [segmentEnv, zeroEnv, HeapCalls.setLocal, count])
          (by simp [nextEnv, HeapCalls.setLocal]) (by simpa [nextEnv, HeapCalls.setLocal] using result)
          (le_trans nonnegative (le_max_left _ _)) positive stopBound
        rw [split, ← max_assoc]
        apply HeapCalls.BlockReturns.branch (value := .scalar (.bool false)) (flag := false) (middle := heap)
        · exact HeapCalls.Evaluates.compare (av := .scalar (.int first)) (bv := .scalar (.int digits.length))
            (HeapCalls.Evaluates.lookup (by simp [HeapCalls.setLocal]))
            (HeapCalls.Evaluates.lookup (by simp [HeapCalls.setLocal, count]))
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
        · rfl
        · apply HeapCalls.BlockReturns.assign (value := .scalar (.int first)) (middle := heap)
            (HeapCalls.Evaluates.lookup (by simp [HeapCalls.setLocal]))
          apply HeapCalls.BlockContinues.returns scanned
          apply HeapCalls.BlockContinues.returns process
          apply HeapCalls.BlockReturns.assign (value := .scalar (.int stop)) (middle := heap)
          · exact HeapCalls.Evaluates.lookup (by
              rw [preserved "j" (by decide) (by decide) (by decide) (by decide)]
              simp [segmentEnv, HeapCalls.setLocal])
          · exact HeapCalls.BlockReturns.loop_resume next
  · have atEnd : start = digits.length := by omega
    apply HeapCalls.BlockReturns.while_loop (value := .scalar (.bool false)) (flag := false) (middle := heap)
    · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup index) (HeapCalls.Evaluates.lookup count)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, valid])
    · rfl
    · simpa [atEnd, maximumWindow, maximum, windowValues, max_eq_left nonnegative] using
        HeapCalls.BlockReturns.return_value (module := program) (heap := heap) (rest := []) (HeapCalls.Evaluates.lookup previous)
termination_by digits.length - start

/-- The zero-width branch returns before any segment or product loops. -/
theorem solve.zero (digits : List Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (ints digits))) :
    HeapCalls.ReturnsValue program "solve" [.ref address, .scalar (.int 0)] heap
      (.scalar (.int (if digits.isEmpty then 0 else 1))) heap := by
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.assign (value := .scalar (.int digits.length)) (middle := heap)
  · exact HeapCalls.Evaluates.builtin (.cons (HeapCalls.Evaluates.lookup rfl) .nil)
      (by simp [HeapCalls.primitive, Mutable.length, found, ints])
  · apply HeapCalls.BlockReturns.branch (value := .scalar (.bool true)) (middle := heap) (flag := true)
    · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant rfl
    · rfl
    · apply HeapCalls.BlockReturns.return_value
      apply HeapCalls.Evaluates.conditional (value := .scalar (.bool digits.isEmpty)) (middle := heap) (flag := digits.isEmpty)
      · apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
        cases digits <;> simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        omega
      · rfl
      · cases digits.isEmpty <;> exact HeapCalls.Evaluates.constant

/-- Every nonnegative window width returns the naive maximum without changing the heap. -/
theorem solve.result (digits : List Int) (width : Nat) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (ints digits))) :
    HeapCalls.ReturnsValue program "solve" [.ref address, .scalar (.int width)] heap
      (.scalar (.int (maximumWindow digits width))) heap := by
  by_cases zero : width = 0
  · subst width
    simpa only [maximumWindow_zeroWidth] using solve.zero digits heap address found
  · apply HeapCalls.BlockReturns.function rfl rfl
    apply HeapCalls.BlockReturns.assign (value := .scalar (.int digits.length)) (middle := heap)
    · exact HeapCalls.Evaluates.builtin (.cons (HeapCalls.Evaluates.lookup rfl) .nil)
        (by simp [HeapCalls.primitive, Mutable.length, found, ints])
    · apply HeapCalls.BlockReturns.branch (value := .scalar (.bool false)) (middle := heap) (flag := false)
      · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero])
      · rfl
      · apply HeapCalls.BlockReturns.assign HeapCalls.Evaluates.constant
        apply HeapCalls.BlockReturns.assign HeapCalls.Evaluates.constant
        have normalized : max (0 : Int) (maximumWindow (digits.drop 0) width) = maximumWindow digits width := by
          simpa using max_eq_right (maximumWindow_nonnegative digits width)
        rw [← normalized]
        exact outer_result digits heap address width 0 _ 0 found rfl rfl rfl rfl rfl
          (by omega) (by omega) (by omega)

/-- On natural inputs, reuse the existing mathematical Euler equivalence theorem. -/
theorem solve.naturals (digits : List Nat) (width : Nat) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (ints (digits.map Int.ofNat)))) :
    HeapCalls.ReturnsValue program "solve" [.ref address, .scalar (.int width)] heap
      (.scalar (.int (ProjectEulerSolutions.P8.solve digits width))) heap := by
  obtain ⟨_, _, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p8.naive.result
      (digits.map Int.ofNat) width heap address found
  obtain ⟨_, _, mathematical⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p8.naive.naturals digits width heap address found
  have same := Mutable.Call.unique reference.finishes mathematical.finishes
  have output : maximumWindow (digits.map Int.ofNat) width = (ProjectEulerStatements.P8.naive digits width : Int) := by
    have outcomes := congrArg Mutable.Completion.outcome same
    simpa only [Except.ok.injEq, Mutable.Value.scalar.injEq, Constant.int.injEq] using outcomes
  simpa only [output, ProjectEulerSolutions.P8.equiv] using solve.result (digits.map Int.ofNat) width heap address found

private abbrev conditionEnv (k : Int) : HeapCalls.Env :=
  fun name => if name = "k" then some (.scalar (.int k)) else none

private theorem solve.condition (heap : Mutable.Heap) (k : Int) :
    HeapCalls.Evaluates program (conditionEnv k) Solvers.P8Ast.requires_solve_0 heap
      (.scalar (.bool (decide (0 ≤ k)))) heap := by
  exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

theorem requires_solve_0_supported (heap : Mutable.Heap) (digits : Nat) (k : Int)
    (_valid : Mutable.Heap.Valid heap) (_typed : Mutable.Value.HasType heap (.ref digits) "list[int]") :
    (HeapCalls.evaluate program (conditionEnv k) Solvers.P8Ast.requires_solve_0 heap).Supported :=
  Execution.supported_of_finishes (solve.condition heap k).finishes

theorem solve.«Equiv(naive)» (heap : Mutable.Heap) (digits : Nat) (k : Int)
    (valid : Mutable.Heap.Valid heap) (typed : Mutable.Value.HasType heap (.ref digits) "list[int]")
    (requires : HeapCalls.Condition program (conditionEnv k) Solvers.P8Ast.requires_solve_0 heap) :
    (HeapCalls.call program "solve" [.ref digits, .scalar (.int k)] heap).Equiv
      (HeapCalls.call P8Ast.module "naive" [.ref digits, .scalar (.int k)] heap) := by
  have nonnegative : 0 ≤ k := by simpa using (solve.condition heap k).condition_iff.mp requires
  lift k to Nat using nonnegative
  obtain ⟨values, found⟩ := Mutable.Value.ints_of_hasType typed
  obtain ⟨finalHeap, extension, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p8.naive.result values k heap digits found
  apply Mutable.Call.equiv_of_finishes (solve.result values k heap digits found).finishes reference.finishes
  obtain ⟨extra, rfl⟩ := extension
  simpa only [List.append_nil, maximumWindow, maximum] using
    Mutable.Completion.equivalent_allocations heap [] extra (.scalar (.int (maximumWindow values k))) [] valid
      trivial (by simp)

theorem solve.«Supported» :
    (∀ (heap : Mutable.Heap) (digits : Nat) (k : Int), Mutable.Heap.Valid heap →
      Mutable.Value.HasType heap (.ref digits) "list[int]" →
      HeapCalls.Condition program (conditionEnv k) Solvers.P8Ast.requires_solve_0 heap →
      (HeapCalls.call program "solve" [.ref digits, .scalar (.int k)] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (digits : Nat) (k : Int), Mutable.Heap.Valid heap →
      Mutable.Value.HasType heap (.ref digits) "list[int]" →
      HeapCalls.Condition program (conditionEnv k) Solvers.P8Ast.requires_solve_0 heap →
      (HeapCalls.call P8Ast.module "naive" [.ref digits, .scalar (.int k)] heap).Supported) := by
  constructor
  · intro heap digits k _ typed requires
    have nonnegative : 0 ≤ k := by simpa using (solve.condition heap k).condition_iff.mp requires
    lift k to Nat using nonnegative
    obtain ⟨values, found⟩ := Mutable.Value.ints_of_hasType typed
    exact (solve.result values k heap digits found).returns.supported
  · intro heap digits k valid typed _
    exact data.«project-euler-statements».data.py.project_euler_statements.p8.naive.«Supported» heap digits k valid typed

end solvers.«8»
