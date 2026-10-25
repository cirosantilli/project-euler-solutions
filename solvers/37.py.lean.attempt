import ProjectEulerStatements.Lib.TruncatablePrimesScan
import lang.py.FormalSW.SumCalls
import data.«project-euler-statements».data.py.project_euler_statements.«p37.py»
import «.formalsw».solvers.«37.py.ast»
import ProjectEulerSolutions.Lib.PrimeSieve
import ProjectEulerSolutions.Lib.OddPrimeSieve
import ProjectEulerStatements.P37
import ProjectEulerStatements.Lib.TruncatablePrimesDigits
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.IntegerParsing
import ProjectEulerStatements.Lib.DecimalDigits

set_option Elab.async false

namespace solvers.«37»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P37Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

namespace sieve

private theorem heap_inside {heap : Mutable.Heap} {address : Nat}
    (found : heap[address]? = some (obj : Mutable.Object)) : address < heap.length := by
  by_contra invalid
  rw [List.getElem?_eq_none (by omega)] at found
  cases found

private theorem heap_set_same (heap : Mutable.Heap) (address : Nat) (obj : Mutable.Object)
    (found : heap[address]? = some obj) : heap.set address obj = heap := by
  apply List.ext_getElem?
  intro i
  by_cases same : i = address
  · subst i
    rw [List.getElem?_set_self (heap_inside found), found]
  · simp [List.getElem?_set_ne (Ne.symm same)]

private def zeroBody (name : String) : List Stmt :=
  [.assign [.subscript (.name "is_prime") (.name name)] (.constant (.int 0))]

private def zeroes (bytes : List UInt8) (first count step : Nat) : List UInt8 :=
  (List.range' first count step).foldl (fun result i => result.set i 0) bytes

private theorem zeroes_length (bytes : List UInt8) (first count step : Nat) :
    (zeroes bytes first count step).length = bytes.length := by
  unfold zeroes
  generalize List.range' first count step = indices
  induction indices generalizing bytes with
  | nil => rfl
  | cons i rest ih => simpa only [List.foldl_cons, List.length_set] using ih (bytes.set i 0)

private theorem zero_write (name : String) (index address : Nat) (bytes : List UInt8)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (inside : index < bytes.length)
    (getIndex : env name = some (number index))
    (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    BlockContinues program env (zeroBody name) heap env
      (heap.set address (.bytearray (bytes.set index 0))) := by
  apply BlockContinues.setItem (value := number 0) (by intros; simp) Evaluates.constant
    (.cons (Evaluates.lookup getBuffer) (.cons (Evaluates.lookup getIndex) .nil))
  exact Mutable.setItem_buffer heap address index bytes 0 found inside

/-- Both source marking loops use the same range semantics and mutate the existing buffer. -/
private theorem zero_loop (name : String) (distinct : name ≠ "is_prime")
    (first count stop step address : Nat) (bytes : List UInt8)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (positive : 0 < step)
    (endBound : stop ≤ first + count * step)
    (lastBound : count = 0 ∨ first + (count-1) * step < stop)
    (size : stop ≤ bytes.length)
    (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop name first stop step (zeroBody name) []] heap finalEnv
      (heap.set address (.bytearray (zeroes bytes first count step))) ∧
      ∀ query, query ≠ name → finalEnv query = env query := by
  induction count generalizing first bytes heap env with
  | zero =>
      refine ⟨env, ?_, fun _ _ => rfl⟩
      simp only [zeroes, List.range'_zero, List.foldl_nil, heap_set_same heap address _ found]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      have last : first + count * step < stop := by simpa using lastBound.resolve_left (by omega)
      have inside : first < stop := by nlinarith
      let bound := setLocal env name (number first)
      let middle := heap.set address (.bytearray (bytes.set first 0))
      have boundBuffer : bound "is_prime" = some (.ref address) := by
        simpa [bound, setLocal, Ne.symm distinct] using getBuffer
      have written := zero_write name first address bytes heap bound (by omega) (by simp [bound, setLocal]) boundBuffer found
      have nextFound : middle[address]? = some (.bytearray (bytes.set first 0)) :=
        List.getElem?_set_self (heap_inside found)
      have nextEnd : stop ≤ (first+step) + count*step := by nlinarith
      have nextLast : count = 0 ∨ (first+step) + (count-1)*step < stop := by
        by_cases empty : count = 0
        · exact Or.inl empty
        · right
          have : count-1+1 = count := by omega
          nlinarith
      obtain ⟨finalEnv, searched, stable⟩ := ih (first+step) (bytes.set first 0) middle bound
        nextEnd nextLast (by simpa using size) boundBuffer nextFound
      refine ⟨finalEnv, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          zeroes, List.range'_succ, middle, List.set_set, Nat.cast_add] using written.append searched
      · intro query different
        exact (stable query different).trans (by simp [bound, setLocal, different])

end sieve

namespace is_truncatable_prime
open ProjectEulerStatements
private abbrev decimal (n : Nat) := toString (n : Int)
private abbrev text (n : Nat) : Mutable.Value := .scalar (.string (decimal n))

private theorem decimal_characters (n : Nat) (positive : 0 < n) :
    (decimal n).toList = (P37.digitsBE n).map Nat.digitChar := by
  simpa [decimal, toString, Int.repr, Nat.repr, P37.digitsBE] using
    DecimalDigits.toDigits_eq_digitChars n positive

private theorem decimal_length (n : Nat) (positive : 0 < n) :
    (decimal n).toList.length = (P37.digitsBE n).length := by
  rw [decimal_characters n positive, List.length_map]

private theorem digit_bounds (n : Nat) : ∀ d ∈ P37.digitsBE n, d < 10 := by
  intro d member
  exact Nat.digits_lt_base (by decide) (List.mem_reverse.mp member)

private theorem parse_digits (digits : List Nat) (nonempty : digits ≠ [])
    (valid : ∀ d ∈ digits, d < 10) :
    Calls.integerConversion [.string (String.ofList (digits.map Nat.digitChar))] =
      some (.ok (.int (P37.digitsToNat digits))) := by
  have parsed := FormalSW.Py.Builtin.Int.parseText_digits (digits.map Nat.digitChar)
    (by simpa using nonempty) (by
      intro c member
      obtain ⟨d, digitMember, rfl⟩ := List.mem_map.mp member
      simpa only [Nat.isDigit_digitChar, decide_eq_true_eq] using valid d digitMember)
  have value : Nat.ofDigitChars 10 (digits.map Nat.digitChar) 0 = P37.digitsToNat digits := by
    simpa [Nat.ofDigitChars, Nat.mul_comm, P37.digitsToNat] using
      DecimalDigits.foldl_digitChars digits valid
  rw [value] at parsed
  rw [Calls.integerConversion, parsed]
  rfl

private theorem substring_int (digits : List Nat) (nonempty : digits ≠ [])
    (valid : ∀ d ∈ digits, d < 10) (expression : Expr)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (evaluated : Evaluates program env expression heap
      (.scalar (.string (String.ofList (digits.map Nat.digitChar)))) heap) :
    Evaluates program env (.builtinCall .int [expression]) heap
      (number (P37.digitsToNat digits)) heap := by
  apply Evaluates.builtin (.cons evaluated .nil)
  simp only [primitive, List.mapM_cons, List.mapM_nil, pure_bind]
  rw [parse_digits digits nonempty valid]
  rfl

private theorem left_slice (n i : Nat) (positive : 0 < n)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getS : env "s" = some (text n)) (getI : env "i" = some (number i)) :
    Evaluates program env (.subscript (.name "s") (.slice (some (.name "i")) none none)) heap
      (.scalar (.string (String.ofList (((P37.digitsBE n).drop i).map Nat.digitChar)))) heap := by
  apply Evaluates.slice (.cons (Evaluates.lookup getS) (.cons (Evaluates.lookup getI)
    (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
  change (match Strings.slice (.string (decimal n)) (.int i) .none .none with
    | .ok value => some (Except.ok (Mutable.Value.scalar value), heap)
    | .error error => some (Except.error error, heap)) = _
  have sliced : Strings.slice (.string (decimal n)) (.int i) .none .none =
      .ok (.string (String.ofList ((decimal n).toList.drop i))) := by
    change (match FormalSW.Py.Builtin.Sequence.slice (decimal n).toList { start := some i } with
      | .ok chars => Except.ok (Constant.string (String.ofList chars))
      | .error _ => Except.error ({ typeName := "ValueError", args := [] } : ExceptionObject)) = _
    rw [FormalSW.Py.Builtin.Sequence.slice_suffix]
  rw [sliced, decimal_characters n positive, ← List.map_drop]

private theorem right_slice (n i : Nat) (positive : 0 < n) (inside : i ≤ (P37.digitsBE n).length)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getS : env "s" = some (text n)) (getI : env "i" = some (number i)) :
    Evaluates program env (.subscript (.name "s") (.slice none
      (some (.binOp (.builtinCall .len [.name "s"]) .sub (.name "i"))) none)) heap
      (.scalar (.string (String.ofList (((P37.digitsBE n).take ((P37.digitsBE n).length-i)).map Nat.digitChar)))) heap := by
  have lengthResult : Evaluates program env (.builtinCall .len [.name "s"]) heap
      (number (P37.digitsBE n).length) heap := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getS) .nil)
    have equal := DecimalDigits.int_string_length n positive
    simpa [primitive, Mutable.length, number, text, decimal, P37.digitsBE, equal]
  have difference : Evaluates program env (.binOp (.builtinCall .len [.name "s"]) .sub (.name "i"))
      heap (number ((P37.digitsBE n).length-i : Nat)) heap :=
    Evaluates.binary lengthResult (Evaluates.lookup getI)
      (by simp [Mutable.binary, scalarBinary, number, Nat.cast_sub inside])
  apply Evaluates.slice (.cons (Evaluates.lookup getS) (.cons Evaluates.constant
    (.cons difference (.cons Evaluates.constant .nil))))
  let count := (P37.digitsBE n).length-i
  have countBound : count ≤ (decimal n).toList.length := by rw [decimal_length n positive]; omega
  have normalized : FormalSW.Py.Builtin.Slice.prefixLength (decimal n).toList.length count = count := by
    rw [FormalSW.Py.Builtin.Slice.prefixLength_nonnegative, min_eq_left countBound]
  have sliced : Strings.slice (.string (decimal n)) .none (.int count) .none =
      .ok (.string (String.ofList ((decimal n).toList.take count))) := by
    change (match FormalSW.Py.Builtin.Sequence.slice (decimal n).toList { stop := some count } with
      | .ok chars => Except.ok (Constant.string (String.ofList chars))
      | .error _ => Except.error ({ typeName := "ValueError", args := [] } : ExceptionObject)) = _
    rw [FormalSW.Py.Builtin.Sequence.slice_prefix, normalized]
  change (match Strings.slice (.string (decimal n)) .none (.int count) .none with
    | .ok value => some (Except.ok (Mutable.Value.scalar value), heap)
    | .error error => some (Except.error error, heap)) = _
  rw [sliced, decimal_characters n positive, ← List.map_take]

end is_truncatable_prime

namespace sieve
open ProjectEulerSolutions.PrimeSieve ProjectEulerSolutions.OddPrimeSieve

private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "sieve").getD ([], [])).2
private abbrev whileCondition : Expr :=
  .compare (.binOp (.name "p") .mult (.name "p")) [(.ltE, .name "limit")]
private abbrev whileBody : List Stmt :=
  match (code[5]? : Option Stmt) with | some (.whileDo _ body _) => body | _ => []
private abbrev markBody : List Stmt :=
  match (whileBody[0]? : Option Stmt) with | some (.ifThen _ body _) => body | _ => []

private def Invariant (limit p : Nat) (bytes : List UInt8) : Prop :=
  Sieved limit p bytes ∧ Binary bytes ∧ ∀ i < 2, bytes[i]? = some 0
private def Stable (before after : HeapCalls.Env) : Prop :=
  after "limit" = before "limit" ∧ after "is_prime" = before "is_prime"

private theorem read_buffer (heap : Mutable.Heap) (env : HeapCalls.Env) (address p : Nat)
    (bytes : List UInt8) (valid : p < bytes.length)
    (getP : env "p" = some (number p)) (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    Evaluates program env (.subscript (.name "is_prime") (.name "p")) heap
      (number bytes[p].toNat) heap := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro h; cases h) (Evaluates.lookup getBuffer) (Evaluates.lookup getP)
  exact Mutable.getItem_buffer heap address p bytes found valid

private theorem mark_result (limit p address : Nat) (bytes : List UInt8)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (lower : 3 ≤ p)
    (size : bytes.length = limit+1)
    (getP : env "p" = some (number p)) (getLimit : env "limit" = some (number limit))
    (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    ∃ finalEnv, BlockContinues program env markBody heap finalEnv
      (heap.set address (.bytearray (oddMark bytes p))) ∧
      Stable env finalEnv ∧ finalEnv "p" = env "p" := by
  let step := 2*p
  let count := (limit+1-p*p+step-1)/step
  let e1 := setLocal env "step" (number step)
  let e2 := setLocal e1 "start" (number (p*p : Nat))
  have bounds := zeroRange_bounds (p*p) (limit+1) step (by dsimp [step]; omega)
  obtain ⟨finalEnv, marked, saved⟩ := zero_loop "x" (by decide) (p*p) count (limit+1) step address
    bytes heap e2 (by dsimp [step]; omega) bounds.1 bounds.2 (by omega)
    (by simpa [e2, e1, setLocal] using getBuffer) found
  refine ⟨finalEnv, ?_, ⟨?_, ?_⟩, ?_⟩
  · have first : BlockContinues program env [markBody[0]'(by decide)] heap e1 heap := by
      apply BlockContinues.assign
      exact Evaluates.binary Evaluates.constant (Evaluates.lookup getP)
        (by simp [Mutable.binary, scalarBinary, number, step])
    have second : BlockContinues program e1 [markBody[1]'(by decide)] heap e2 heap := by
      apply BlockContinues.assign
      exact Evaluates.binary (Evaluates.lookup (by simpa [e1, setLocal] using getP))
        (Evaluates.lookup (by simpa [e1, setLocal] using getP))
        (by simp [Mutable.binary, scalarBinary, number])
    have third : BlockContinues program e2 [markBody[2]'(by decide)] heap finalEnv
        (heap.set address (.bytearray (oddMark bytes p))) := by
      apply BlockContinues.for_range (start := (p*p : Nat)) (stop := (limit+1 : Nat))
        (stride := step) (middle := heap)
        (.cons (Evaluates.lookup rfl) (.cons
        (Evaluates.binary (value := number (limit+1 : Nat)) (finalHeap := heap) (Evaluates.lookup (by simpa [e2, e1, setLocal] using getLimit))
          Evaluates.constant (by simp [Mutable.binary, scalarBinary, number]))
        (.cons (Evaluates.lookup (value := number step) (by simp [e2, e1, setLocal])) .nil)))
        (by
          change (if (step : Int) = 0 then
            Except.error ({ typeName := "ValueError", args := [] } : ExceptionObject)
            else Except.ok (((p*p : Nat) : Int), ((limit+1 : Nat) : Int), (step : Int))) = _
          rw [if_neg (show (step : Int) ≠ 0 by dsimp [step]; omega)])
      simpa only [oddMark, zeroRange, zeroes, size, count, step] using marked
    exact first.append (second.append third)
  · exact (saved "limit" (by decide)).trans (by simp [e2, e1, setLocal])
  · exact (saved "is_prime" (by decide)).trans (by simp [e2, e1, setLocal])
  · exact (saved "p" (by decide)).trans (by simp [e2, e1, setLocal])

private theorem iteration (limit p address : Nat) (bytes : List UInt8)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (lower : 3 ≤ p) (odd : p%2=1)
    (inside : p*p ≤ limit) (getP : env "p" = some (number p))
    (getLimit : env "limit" = some (number limit)) (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (invariant : Invariant limit p bytes) :
    ∃ finalEnv finalBytes, BlockContinues program env whileBody heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Invariant limit (p+2) finalBytes ∧
      Stable env finalEnv ∧ finalEnv "p" = some (number (p+2 : Nat)) := by
  have size := invariant.1.1
  have valid : p < bytes.length := by rw [size]; nlinarith
  have live : bytes[p].toNat ≠ 0 ↔ Nat.Prime p :=
    (invariant.1.2 p bytes[p] (by omega) (List.getElem?_eq_getElem valid)).trans
      (survives_current p (by omega))
  have evaluated := read_buffer heap env address p bytes valid getP getBuffer found
  have branch : ∃ middleEnv middleBytes,
      BlockContinues program env [whileBody[0]'(by decide)] heap middleEnv
        (heap.set address (.bytearray middleBytes)) ∧ Invariant limit (p+2) middleBytes ∧
        Stable env middleEnv ∧ middleEnv "p" = some (number p) := by
    by_cases prime : Nat.Prime p
    · obtain ⟨middleEnv, marked, saved, sameP⟩ := mark_result limit p address bytes env heap lower
        size getP getLimit getBuffer found
      refine ⟨middleEnv, oddMark bytes p,
        BlockContinues.branch (flag := true) evaluated (by simp [Mutable.truth, scalarTruth, live.mpr prime]) marked,
        ⟨sieved_oddMark limit p bytes invariant.1 lower prime, oddMark_binary bytes p invariant.2.1, ?_⟩,
        saved, sameP.trans getP⟩
      intro i small
      rw [oddMark_get_low bytes p i lower small]
      exact invariant.2.2 i small
    · refine ⟨env, bytes, ?_,
        ⟨sieved_oddSkip limit p bytes invariant.1 lower odd prime, invariant.2⟩,
        ⟨rfl, rfl⟩, getP⟩
      rw [heap_set_same heap address _ found]
      have zero : bytes[p].toNat = 0 := by tauto
      exact BlockContinues.branch (flag := false) evaluated (by simp [Mutable.truth, scalarTruth, zero]) .nil
  obtain ⟨middleEnv, middleBytes, branched, nextInvariant, saved, sameP⟩ := branch
  let updated := setLocal middleEnv "p" (number (p+2 : Nat))
  refine ⟨updated, middleBytes, ?_, nextInvariant, ?_, by simp [updated, setLocal]⟩
  · have increment : BlockContinues program middleEnv [whileBody[1]'(by decide)]
        (heap.set address (.bytearray middleBytes)) updated (heap.set address (.bytearray middleBytes)) :=
      BlockContinues.augAssign sameP Evaluates.constant (by simp [inplace, Mutable.binary, scalarBinary, number])
    exact branched.append increment
  · simpa [Stable, updated, setLocal] using saved

private theorem loop (fuel limit p address : Nat) (bytes : List UInt8)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (lower : 3 ≤ p) (odd : p%2=1)
    (enough : limit+1 < p+fuel) (getP : env "p" = some (number p))
    (getLimit : env "limit" = some (number limit)) (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (invariant : Invariant limit p bytes) :
    ∃ finalEnv finalBytes finalP,
      BlockContinues program env [.whileDo whileCondition whileBody []] heap finalEnv
        (heap.set address (.bytearray finalBytes)) ∧ Invariant limit finalP finalBytes ∧
        limit < finalP^2 ∧ Stable env finalEnv := by
  induction fuel generalizing p bytes env heap with
  | zero =>
      have finished : ¬ p*p ≤ limit := by nlinarith
      have tested : Evaluates program env whileCondition heap (.scalar (.bool false)) heap := by
        apply Evaluates.compare (av := number (p*p : Nat)) (bv := number limit)
          (middle := heap) (afterRight := heap)
          (Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP)
          (by simp [Mutable.binary, scalarBinary, number])) (Evaluates.lookup getLimit)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        exact_mod_cast (Nat.lt_of_not_ge finished)
      refine ⟨env, bytes, p, ?_, invariant, by nlinarith, rfl, rfl⟩
      rw [heap_set_same heap address _ found]
      exact BlockContinues.while_loop tested rfl .nil
  | succ fuel ih =>
      have tested : Evaluates program env whileCondition heap (.scalar (.bool (decide (p*p≤limit)))) heap := by
        apply Evaluates.compare (av := number (p*p : Nat)) (bv := number limit)
          (middle := heap) (afterRight := heap)
          (Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP)
          (by simp [Mutable.binary, scalarBinary, number])) (Evaluates.lookup getLimit)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        exact_mod_cast (Iff.rfl : p*p ≤ limit ↔ p*p ≤ limit)
      by_cases inside : p*p≤limit
      · obtain ⟨middleEnv, middleBytes, iterated, nextInvariant, saved, nextP⟩ :=
          iteration limit p address bytes env heap lower odd inside getP getLimit getBuffer found invariant
        let middleHeap := heap.set address (.bytearray middleBytes)
        obtain ⟨finalEnv, finalBytes, finalP, rest, finalInvariant, finished, stable⟩ :=
          ih (p+2) middleBytes middleEnv middleHeap (by omega) (by omega) (by omega) nextP
            (saved.1.trans getLimit) (saved.2.trans getBuffer)
            (List.getElem?_set_self (heap_inside found)) nextInvariant
        refine ⟨finalEnv, finalBytes, finalP, ?_, finalInvariant, finished,
          stable.1.trans saved.1, stable.2.trans saved.2⟩
        apply BlockContinues.while_loop (flag := true) (by simpa [inside] using tested) rfl
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, middleHeap, List.set_set]
          using iterated.append rest
      · refine ⟨env, bytes, p, ?_, invariant, by nlinarith, rfl, rfl⟩
        rw [heap_set_same heap address _ found]
        exact BlockContinues.while_loop (flag := false) (by simpa [inside] using tested) rfl .nil

end sieve

namespace sieve
open ProjectEulerSolutions.PrimeSieve ProjectEulerSolutions.OddPrimeSieve

private def initialHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1], .bytearray (initialEven limit)]

private theorem replace_initial (limit : Nat) (large : 2 ≤ limit) :
    FormalSW.Py.Builtin.ByteArray.setSlice (List.replicate (limit+1) 1)
      { start := some 0, stop := some 2 } [0,0] = .ok (initial limit) := by
  have normalized : FormalSW.Py.Builtin.Slice.indices
      { start := some 0, stop := some 2 } (limit+1) = .ok (0,2,1) := by
    change (Except.ok
      (max 0 (min 0 ((limit+1 : Nat) : Int)), max 0 (min 2 ((limit+1 : Nat) : Int)), (1 : Int)) :
      Except FormalSW.Py.Builtin.Error (Int × Int × Int)) = _
    rw [min_eq_left (show (0 : Int) ≤ (limit+1 : Nat) by omega),
      min_eq_left (show (2 : Int) ≤ (limit+1 : Nat) by omega)]
    rfl
  change FormalSW.Py.Builtin.List.setSlice (List.replicate (limit+1) 1)
    { start := some 0, stop := some 2 } [0,0] = _
  unfold FormalSW.Py.Builtin.List.setSlice
  rw [List.length_replicate, normalized]
  change (Except.ok ((List.replicate (limit+1) (1 : UInt8)).take 0 ++ [0,0] ++
    (List.replicate (limit+1) 1).drop 2) : Except FormalSW.Py.Builtin.Error (List UInt8)) = _
  simp [initial, show limit+1-2 = limit-1 by omega]

private theorem setup (limit : Nat) (large : 2 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (number limit)) :
    ∃ finalEnv, BlockContinues program env ((code.drop 1).take 4) heap finalEnv (initialHeap heap limit) ∧
      finalEnv "p" = some (number 3) ∧ finalEnv "limit" = env "limit" ∧
      finalEnv "is_prime" = some (.ref (heap.length+1)) := by
  let e1 := setLocal env "is_prime" (.ref (heap.length+1))
  let allocated := heap ++ [.bytearray [1], .bytearray (List.replicate (limit+1) 1)]
  let cleared := heap ++ [.bytearray [1], .bytearray (initial limit)]
  have first : BlockContinues program env [code[1]'(by decide)] heap e1 allocated := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := .ref heap.length) (middle := heap ++ [.bytearray [1]])
      (bv := number (limit+1 : Nat)) (afterRight := heap ++ [.bytearray [1]])
    · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
    · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])
    · simp [allocated, Mutable.binary, Mutable.allocateBuffer, Except.map,
        show (BinOp.mult == .mult) = true from rfl]
  have second : BlockContinues program e1 [code[2]'(by decide)] allocated e1 cleared := by
    apply BlockContinues.setSlice (value := .scalar (.bytes [0,0]))
      (afterRhs := allocated) (afterTarget := allocated) (receiver := .ref (heap.length+1))
      (lowerValue := number 0) (upperValue := number 2) (strideValue := .scalar .none)
      Evaluates.constant
      (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
    have updated := Mutable.setSlice_buffer allocated (heap.length+1) (List.replicate (limit+1) 1)
      [0,0] (initial limit) (number 0) (number 2) (.scalar .none)
      { start := some 0, stop := some 2 } (by simp [allocated]) rfl (replace_initial limit large)
    simpa [allocated, cleared] using updated
  let count := (limit+1-4+2-1)/2
  have bounds := zeroRange_bounds 4 (limit+1) 2 (by decide)
  obtain ⟨markedEnv, marked, saved⟩ := zero_loop "i" (by decide) 4 count (limit+1) 2 (heap.length+1)
    (initial limit) cleared e1 (by decide) bounds.1 bounds.2 (by have := (sieved_initial limit large).1; omega)
    rfl (by simp [cleared])
  have third : BlockContinues program e1 [code[3]'(by decide)] cleared markedEnv (initialHeap heap limit) := by
    apply BlockContinues.for_range (start := 4) (stop := (limit+1 : Nat)) (stride := 2) (middle := cleared)
      (.cons Evaluates.constant (.cons
      (Evaluates.binary (value := number (limit+1 : Nat)) (finalHeap := cleared) (Evaluates.lookup (by simpa [e1, setLocal] using getLimit)) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])) (.cons Evaluates.constant .nil))) rfl
    simpa [initialEven, zeroRange, zeroes, initialHeap, cleared, count] using marked
  let updated := setLocal markedEnv "p" (number 3)
  have fourth : BlockContinues program markedEnv [code[4]'(by decide)] (initialHeap heap limit)
      updated (initialHeap heap limit) := BlockContinues.assign Evaluates.constant
  refine ⟨updated, first.append (second.append (third.append fourth)), ?_, ?_, ?_⟩
  · simp [updated, setLocal]
  · simpa [updated, e1, setLocal] using saved "limit" (by decide)
  · simpa [updated, e1, setLocal] using saved "is_prime" (by decide)

/-- The returned bytearray has the exact zero/one primality flag at every valid index. -/
theorem result (limit : Nat) (large : 2 ≤ limit) (heap : Mutable.Heap) :
    ∃ bytes, ReturnsValue program "sieve" [number limit] heap (.ref (heap.length+1))
      (heap ++ [.bytearray [1], .bytearray bytes]) ∧ bytes.length = limit+1 ∧
      ∀ i ≤ limit, bytes[i]? = some (if Nat.Prime i then 1 else 0) := by
  let env := functionEnv ((findFunctionIn program.body "sieve").getD ([], [])).1
    ((findFunctionIn program.body "sieve").getD ([], [])).2 [number limit] []
  obtain ⟨readyEnv, initialized, getP, sameLimit, getBuffer⟩ := setup limit large env heap rfl
  have invariant : Invariant limit 3 (initialEven limit) :=
    ⟨sieved_initialEven limit large, initialEven_binary limit, initialEven_get_low limit⟩
  obtain ⟨finalEnv, bytes, finalP, executed, finalInvariant, finished, saved⟩ :=
    loop (limit+1) limit 3 (heap.length+1) (initialEven limit) readyEnv (initialHeap heap limit)
      (by decide) rfl (by omega) getP (sameLimit.trans rfl) getBuffer (by simp [initialHeap]) invariant
  refine ⟨bytes, ?_, finalInvariant.1.1, fun i valid =>
    sieved_finished_get limit finalP i bytes finalInvariant.1 finalInvariant.2.1 finalInvariant.2.2 finished valid⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) (middle := heap) (value := .scalar (.bool false))
  · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
  · rfl
  · apply BlockContinues.returns initialized
    have executed' : BlockContinues program readyEnv [code[5]'(by decide)] (initialHeap heap limit)
        finalEnv (heap ++ [.bytearray [1], .bytearray bytes]) := by
      simpa [initialHeap] using executed
    apply BlockContinues.returns executed'
    exact BlockReturns.return_value (Evaluates.lookup (saved.2.trans getBuffer))

end sieve

namespace is_truncatable_prime
open ProjectEulerStatements ProjectEulerStatements.TruncatablePrimes
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "is_truncatable_prime").getD ([], [])).2
private abbrev body : List Stmt :=
  match (code[2]? : Option Stmt) with | some (.forIn _ _ statements _) => statements | _ => []
private abbrev leftExpression : Expr :=
  .builtinCall .int [.subscript (.name "s") (.slice (some (.name "i")) none none)]
private abbrev rightExpression : Expr := .builtinCall .int [.subscript (.name "s") (.slice none
  (some (.binOp (.builtinCall .len [.name "s"]) .sub (.name "i"))) none)]
private def pairCheck (n i : Nat) : Bool :=
  P37.isPrime (P37.digitsToNat ((P37.digitsBE n).drop i)) &&
    P37.isPrime (P37.digitsToNat ((P37.digitsBE n).take ((P37.digitsBE n).length-i)))

private theorem not_prime (index address : Nat) (bytes : List UInt8)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (expression : Expr)
    (ordinary : ∀ lower upper stride, expression ≠ .slice lower upper stride)
    (inside : index < bytes.length) (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes))
    (correct : ∀ i < bytes.length, bytes[i]? = some (if Nat.Prime i then 1 else 0))
    (evaluated : Evaluates program env expression heap (number index) heap) :
    Evaluates program env (.unaryOp .not (.subscript (.name "is_prime") expression)) heap
      (.scalar (.bool (!P37.isPrime index))) heap := by
  have value : bytes[index] = if Nat.Prime index then 1 else 0 := by
    have same := correct index inside
    simpa only [List.getElem?_eq_getElem inside, Option.some.injEq] using same
  apply Evaluates.unary (value := number bytes[index].toNat) (middle := heap)
  · apply Evaluates.subscript (middle := heap) (afterIndex := heap)
      ordinary (Evaluates.lookup getBuffer) evaluated
    exact Mutable.getItem_buffer heap address index bytes found inside
  · rw [value]
    by_cases prime : Nat.Prime index <;>
      simp [HeapCalls.unary, Mutable.truth, scalarTruth, P37.isPrime, prime]

private theorem tests (n i address : Nat) (bytes : List UInt8)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (positive : 0 < n)
    (lower : 0 < i) (upper : i < (P37.digitsBE n).length) (inside : n < bytes.length)
    (getS : env "s" = some (text n)) (getI : env "i" = some (number i))
    (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes))
    (correct : ∀ j < bytes.length, bytes[j]? = some (if Nat.Prime j then 1 else 0)) :
    Evaluates program env (.unaryOp .not (.subscript (.name "is_prime") leftExpression)) heap
      (.scalar (.bool (!P37.isPrime (P37.digitsToNat ((P37.digitsBE n).drop i))))) heap ∧
    Evaluates program env (.unaryOp .not (.subscript (.name "is_prime") rightExpression)) heap
      (.scalar (.bool (!P37.isPrime (P37.digitsToNat ((P37.digitsBE n).take ((P37.digitsBE n).length-i)))))) heap := by
  constructor
  · apply not_prime (P37.digitsToNat ((P37.digitsBE n).drop i)) address bytes env heap _ (by intros; intro h; cases h)
      (by have := left_truncation_lt n i positive lower; omega) getBuffer found correct
    apply substring_int ((P37.digitsBE n).drop i) (fun empty => (Nat.not_le.mpr upper) (List.drop_eq_nil_iff.mp empty))
      (fun d member => digit_bounds n d (List.mem_of_mem_drop member))
    exact left_slice n i positive env heap getS getI
  · apply not_prime (P37.digitsToNat ((P37.digitsBE n).take ((P37.digitsBE n).length-i))) address bytes env heap _ (by intros; intro h; cases h)
      (by have := right_truncation_lt n i positive lower (by omega); omega) getBuffer found correct
    apply substring_int ((P37.digitsBE n).take ((P37.digitsBE n).length-i)) (by
      intro empty
      have length := congrArg List.length empty
      simp only [List.length_take, List.length_nil] at length
      omega) (fun d member => digit_bounds n d (List.mem_of_mem_take member))
    exact right_slice n i positive (by omega) env heap getS getI

private theorem loop (n first count address : Nat) (bytes : List UInt8)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (positive : 0 < n)
    (lower : 0 < first) (endpoint : first+count = (P37.digitsBE n).length)
    (inside : n < bytes.length) (getS : env "s" = some (text n))
    (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes))
    (correct : ∀ j < bytes.length, bytes[j]? = some (if Nat.Prime j then 1 else 0)) :
    BlockReturns program env
      [.rangeStepLoop "i" first (P37.digitsBE n).length 1 body [], .return (.constant (.bool true))]
      heap (.scalar (.bool ((List.range' first count).all (pairCheck n)))) heap := by
  induction count generalizing first env with
  | zero =>
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value Evaluates.constant
  | succ count ih =>
      let bound := setLocal env "i" (number first)
      have boundS : bound "s" = some (text n) := by simpa [bound, setLocal] using getS
      have boundBuffer : bound "is_prime" = some (.ref address) := by simpa [bound, setLocal] using getBuffer
      obtain ⟨left, right⟩ := tests n first address bytes bound heap positive lower (by omega) inside
        boundS (by simp [bound, setLocal]) boundBuffer found correct
      have rest := ih (first+1) bound (by omega) (by omega) boundS boundBuffer
      have next : BlockReturns program bound
          [Calls.loopContinuation body (.rangeStepLoop "i" ((first:Int)+1) (P37.digitsBE n).length 1 body []),
           .return (.constant (.bool true))] heap
          (.scalar (.bool ((List.range' (first+1) count).all (pairCheck n)))) heap := by
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      cases leftPrime : P37.isPrime (P37.digitsToNat ((P37.digitsBE n).drop first)) with
      | false =>
          simp only [List.range'_succ, List.all_cons, pairCheck, leftPrime, Bool.false_and]
          apply BlockReturns.branch (flag := true) (by simpa [leftPrime] using left) rfl
          exact BlockReturns.return_value Evaluates.constant
      | true =>
          simp only [List.range'_succ, List.all_cons, pairCheck, leftPrime, Bool.true_and]
          apply BlockReturns.branch (flag := false) (by simpa [leftPrime] using left) rfl
          cases rightPrime : P37.isPrime (P37.digitsToNat ((P37.digitsBE n).take ((P37.digitsBE n).length-first))) with
          | false =>
              simp only [Bool.false_and]
              apply BlockReturns.branch (flag := true) (by simpa [rightPrime] using right) rfl
              exact BlockReturns.return_value Evaluates.constant
          | true =>
              simp only [Bool.true_and]
              apply BlockReturns.branch (flag := false) (by simpa [rightPrime] using right) rfl
              exact next

/-- The helper checks proper truncations only; its caller checks primality of n itself. -/
theorem result (n address : Nat) (bytes : List UInt8) (heap : Mutable.Heap)
    (inside : n < bytes.length) (found : heap[address]? = some (.bytearray bytes))
    (correct : ∀ i < bytes.length, bytes[i]? = some (if Nat.Prime i then 1 else 0)) :
    ReturnsValue program "is_truncatable_prime" [number n, .ref address] heap
      (.scalar (.bool (properTruncatable n))) heap := by
  let env := functionEnv ((findFunctionIn program.body "is_truncatable_prime").getD ([], [])).1
    ((findFunctionIn program.body "is_truncatable_prime").getD ([], [])).2 [number n, .ref address] []
  have tested : Evaluates program env (.compare (.name "n") [(.lt, .constant (.int 10))]) heap
      (.scalar (.bool (decide (n<10)))) heap :=
    Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  apply BlockReturns.function rfl rfl
  by_cases small : n<10
  · simp only [properTruncatable, if_pos small]
    apply BlockReturns.branch (flag := true) (by simpa [small] using tested) rfl
    exact BlockReturns.return_value Evaluates.constant
  · simp only [properTruncatable, if_neg small]
    have positive : 0<n := by omega
    let ready := setLocal env "s" (text n)
    apply BlockReturns.branch (flag := false) (by simpa [small] using tested) rfl
    apply BlockReturns.assign (value := text n) (middle := heap)
      (Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil) rfl)
    have lengthResult : Evaluates program ready (.builtinCall .len [.name "s"]) heap
        (number (P37.digitsBE n).length) heap := by
      apply Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil)
      have equal := DecimalDigits.int_string_length n positive
      simpa [primitive, Mutable.length, number, text, decimal, P37.digitsBE, equal]
    apply BlockReturns.for_range (.cons Evaluates.constant (.cons lengthResult .nil)) rfl
    have lengthPositive : 0 < (P37.digitsBE n).length := by
      apply List.length_pos_iff.mpr
      simpa [P37.digitsBE, Nat.digits_eq_nil_iff_eq_zero] using Nat.ne_of_gt positive
    exact loop n 1 ((P37.digitsBE n).length-1) address bytes ready heap positive (by decide)
      (by omega) inside rfl rfl found correct

end is_truncatable_prime

namespace solve
open ProjectEulerStatements.TruncatablePrimes
private abbrev numbers (items : List Nat) : List Mutable.Value := items.map (fun n : Nat => number n)
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private abbrev loopBody : List Stmt :=
  match (code[9]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
private abbrev candidate : Expr := .boolOp .and
  [.subscript (.name "is_prime") (.name "n"),
    .call (.name "is_truncatable_prime") [.name "n", .name "is_prime"]]
private def collectorHeap (base : Mutable.Heap) (items : List Nat) : Mutable.Heap :=
  base ++ [.list (numbers items)]

private theorem found_append {heap : Mutable.Heap} {address : Nat}
    (found : heap[address]? = some (object : Mutable.Object)) (extra : Mutable.Heap) :
    (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem read_prime (n address : Nat) (bytes : List UInt8) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (index : Expr) (inside : n < bytes.length)
    (found : heap[address]? = some (.bytearray bytes))
    (correct : ∀ i < bytes.length, bytes[i]? = some (if Nat.Prime i then 1 else 0))
    (getSieve : env "is_prime" = some (.ref address))
    (ordinary : ∀ lower upper stride, index ≠ .slice lower upper stride)
    (evaluatedIndex : Evaluates program env index heap (number n) heap) :
    Evaluates program env (.subscript (.name "is_prime") index) heap
      (number (if Nat.Prime n then (1 : Nat) else 0)) heap := by
  have same : bytes[n] = (if Nat.Prime n then 1 else 0) := by
    have checked := correct n inside
    rw [List.getElem?_eq_getElem inside, Option.some.injEq] at checked
    exact checked
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    ordinary (Evaluates.lookup getSieve) evaluatedIndex
  have read := Mutable.getItem_buffer heap address n bytes found inside
  by_cases prime : Nat.Prime n <;> simpa [same, prime] using read

private theorem candidate_result (n address : Nat) (bytes : List UInt8) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (inside : n < bytes.length)
    (found : heap[address]? = some (.bytearray bytes))
    (correct : ∀ i < bytes.length, bytes[i]? = some (if Nat.Prime i then 1 else 0))
    (getSieve : env "is_prime" = some (.ref address)) (getN : env "n" = some (number n)) :
    ∃ value, Evaluates program env candidate heap value heap ∧
      Mutable.truth value heap = some (.ok (ProjectEulerStatements.P37.isTruncatablePrime n), heap) := by
  have read := read_prime n address bytes heap env (.name "n") inside found correct getSieve (by intros; simp) (Evaluates.lookup getN)
  by_cases prime : Nat.Prime n
  · have tested : Evaluates program env (.call (.name "is_truncatable_prime") [.name "n", .name "is_prime"])
        heap (.scalar (.bool (properTruncatable n))) heap :=
      Evaluates.call (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getSieve) .nil))
        (is_truncatable_prime.result n address bytes heap inside found correct)
    refine ⟨.scalar (.bool (properTruncatable n)), ?_, ?_⟩
    · exact Evaluates.boolean_next (flag := true) (by simpa only [if_pos prime] using read) rfl rfl
        (Evaluates.boolean_last tested rfl)
    · simp [Mutable.truth, scalarTruth, reference_eq_prime_and_proper, ProjectEulerStatements.P37.isPrime, prime]
  · refine ⟨number 0, ?_, ?_⟩
    · exact Evaluates.boolean_stop (flag := false) (by simpa only [if_neg prime] using read) rfl rfl
    · simp [Mutable.truth, scalarTruth, reference_eq_prime_and_proper,
        ProjectEulerStatements.P37.isPrime, prime]

/-- The continuation is supplied for the exact list produced by the early-break scan. -/
private theorem scan_loop (first count : Nat) (endpoint : first + 2 * count = 1000001)
    (base : Mutable.Heap) (address : Nat) (bytes : List UInt8) (before : List Nat) (env : HeapCalls.Env)
    (length : bytes.length = 1000001)
    (found : base[address]? = some (.bytearray bytes))
    (correct : ∀ i < bytes.length, bytes[i]? = some (if Nat.Prime i then 1 else 0))
    (getSieve : env "is_prime" = some (.ref address))
    (getCollector : env "truncatable" = some (.ref base.length))
    (following : List Stmt) (answer : Mutable.Value) (finalHeap : Mutable.Heap)
    (finish : ∀ after : HeapCalls.Env, after "truncatable" = some (.ref base.length) →
      BlockReturns program after following
        (collectorHeap base (collectUntilEleven (List.range' first count 2) before)) answer finalHeap) :
    BlockReturns program env
      (.rangeStepLoop "n" first 1000001 2 loopBody [] :: following)
      (collectorHeap base before) answer finalHeap := by
  induction count generalizing first before env with
  | zero =>
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact finish env getCollector
  | succ count ih =>
      let boundEnv := setLocal env "n" (number first)
      have boundSieve : boundEnv "is_prime" = some (.ref address) := by simpa [boundEnv, setLocal] using getSieve
      have boundCollector : boundEnv "truncatable" = some (.ref base.length) := by
        simpa [boundEnv, setLocal] using getCollector
      obtain ⟨value, tested, truth⟩ := candidate_result first address bytes (collectorHeap base before) boundEnv
        (by omega) (found_append found _) correct boundSieve (by simp [boundEnv, setLocal])
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      cases accepted : ProjectEulerStatements.P37.isTruncatablePrime first with
      | false =>
          apply BlockReturns.branch (flag := false) tested (by simpa only [accepted] using truth)
          apply BlockReturns.loop_resume
          have recurse := ih (first + 2) (by omega) before boundEnv boundSieve boundCollector
            (by intro after collector; simpa only [List.range'_succ, collectUntilEleven, accepted, Bool.false_eq_true,
              if_false] using finish after collector)
          simpa only [Nat.cast_add, Nat.cast_ofNat] using recurse
      | true =>
          apply BlockReturns.branch (flag := true) tested (by simpa only [accepted] using truth)
          have appended : Evaluates program boundEnv (.call (.attribute (.name "truncatable") "append") [.name "n"])
              (collectorHeap base before) (.scalar .none) (collectorHeap base (before ++ [first])) := by
            have run := Evaluates.append (globals := []) (module := program) (env := boundEnv)
              (base := .name "truncatable") (expression := .name "n") (value := number first)
              (old := numbers before) (current := numbers before)
              (heap := collectorHeap base before) (Evaluates.lookup boundCollector)
              (by simp [collectorHeap]) (Evaluates.lookup (by simp [boundEnv, setLocal]))
              (by simp [collectorHeap])
            simpa [collectorHeap, numbers, List.set_append_right] using run
          apply BlockReturns.discard appended
          have full : Evaluates program boundEnv
              (.compare (.builtinCall .len [.name "truncatable"]) [(.eq, .constant (.int 11))])
              (collectorHeap base (before ++ [first]))
              (.scalar (.bool (decide ((before ++ [first]).length = 11))))
              (collectorHeap base (before ++ [first])) := by
            apply Evaluates.compare (av := number (before ++ [first]).length) (bv := number 11)
              (middle := collectorHeap base (before ++ [first])) (afterRight := collectorHeap base (before ++ [first]))
              (Evaluates.builtin (.cons (Evaluates.lookup boundCollector) .nil)
                (by simp [primitive, Mutable.length, collectorHeap, numbers])) Evaluates.constant
            simp [HeapCalls.compare, scalarCompare, evalCmpOp]
            omega
          by_cases finished : (before ++ [first]).length = 11
          · apply BlockReturns.branch (flag := true) (by simpa only [finished, decide_true] using full) rfl
            apply BlockReturns.break_loop
              (next := .rangeStepLoop "n" ((first : Int) + 2) 1000001 2 loopBody [])
              (following := following) rfl
            simpa only [List.range'_succ, collectUntilEleven, accepted, ↓reduceIte, finished] using
              finish boundEnv boundCollector
          · apply BlockReturns.branch (flag := false) (by simpa only [finished, decide_false] using full) rfl
            apply BlockReturns.loop_resume
            have recurse := ih (first + 2) (by omega) (before ++ [first]) boundEnv boundSieve boundCollector
              (by intro after collector; simpa only [List.range'_succ, collectUntilEleven, accepted, ↓reduceIte,
                if_neg finished] using finish after collector)
            simpa only [Nat.cast_add, Nat.cast_ofNat] using recurse

private theorem finale (base : Mutable.Heap) (env : HeapCalls.Env)
    (getCollector : env "truncatable" = some (.ref base.length)) :
    BlockReturns program env (code.drop 10) (collectorHeap base values)
      (number values.sum) (collectorHeap base values) := by
  apply BlockReturns.assertion (value := .scalar (.bool true)) (middle := collectorHeap base values)
  · apply Evaluates.compare (av := number 11) (bv := number 11)
      (middle := collectorHeap base values) (afterRight := collectorHeap base values)
      (Evaluates.builtin (.cons (Evaluates.lookup getCollector) .nil)
        (by simp [primitive, Mutable.length, collectorHeap, numbers, values_length])) Evaluates.constant rfl
  · rfl
  · apply BlockReturns.assign (value := number values.sum) (middle := collectorHeap base values)
    · have total : (values.map (fun n : Nat => (n : Int))).foldl Int.add 0 = (values.sum : Int) := by
        change (values.map (fun n : Nat => (n : Int))).foldl (fun a b : Int => a + b) 0 = (values.sum : Int)
        rw [← List.sum_eq_foldl]
        rfl
      change Evaluates program env _ _ (.scalar (.int (values.sum : Int))) _
      rw [← total]
      apply Evaluates.sum_integers (initial := 0) (values := values.map (fun n : Nat => (n : Int)))
      · exact .cons (Evaluates.lookup getCollector) (.cons Evaluates.constant .nil)
      · simp [Mutable.sequence, collectorHeap, numbers, number, List.map_map]
    · exact BlockReturns.return_value (Evaluates.lookup (by simp [setLocal]))

private theorem proper_evaluates (n address : Nat) (bytes : List UInt8) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (inside : n < bytes.length) (found : heap[address]? = some (.bytearray bytes))
    (correct : ∀ i < bytes.length, bytes[i]? = some (if Nat.Prime i then 1 else 0))
    (getSieve : env "is_prime" = some (.ref address)) :
    Evaluates program env (.call (.name "is_truncatable_prime") [.constant (.int n), .name "is_prime"])
      heap (.scalar (.bool (properTruncatable n))) heap :=
  Evaluates.call (.cons Evaluates.constant (.cons (Evaluates.lookup getSieve) .nil))
    (is_truncatable_prime.result n address bytes heap inside found correct)

private theorem not_small_evaluates (n address : Nat) (small : n < 10) (bytes : List UInt8)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (inside : n < bytes.length)
    (found : heap[address]? = some (.bytearray bytes))
    (correct : ∀ i < bytes.length, bytes[i]? = some (if Nat.Prime i then 1 else 0))
    (getSieve : env "is_prime" = some (.ref address)) :
    Evaluates program env (.unaryOp .not
      (.call (.name "is_truncatable_prime") [.constant (.int n), .name "is_prime"]))
      heap (.scalar (.bool true)) heap := by
  apply Evaluates.unary (proper_evaluates n address bytes heap env inside found correct getSieve)
  simp [HeapCalls.unary, Mutable.truth, scalarTruth, properTruncatable, small]

theorem result (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [] heap
      (number (ProjectEulerStatements.P37.naive 1000000)) finalHeap := by
  obtain ⟨bytes, sieved, length, correctBounded⟩ := sieve.result 1000000 (by decide) heap
  let base := heap ++ [.bytearray [1], .bytearray bytes]
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [] []
  let limitEnv := setLocal initial "LIMIT" (number 1000000)
  let sieveEnv := setLocal limitEnv "is_prime" (.ref (heap.length + 1))
  let ready := setLocal sieveEnv "truncatable" (.ref base.length)
  have found : base[heap.length + 1]? = some (.bytearray bytes) := by simp [base]
  have correct : ∀ i < bytes.length, bytes[i]? = some (if Nat.Prime i then 1 else 0) :=
    fun i inside => correctBounded i (by omega)
  have primeExample : Nat.Prime 3797 := reference_prime 3797 (value_accepted 3797 (by decide +kernel))
  have exampleRead := read_prime 3797 (heap.length + 1) bytes base sieveEnv (.constant (.int 3797))
    (by omega) found correct rfl (by intros; simp) Evaluates.constant
  have exampleProper := proper_evaluates 3797 (heap.length + 1) bytes base sieveEnv (by omega) found correct rfl
  have collected : collectUntilEleven (List.range' 11 499995 2) [] = values := collected_values
  have run := scan_loop 11 499995 (by decide) base (heap.length + 1) bytes [] ready length found correct
    (by simp [ready, sieveEnv, setLocal]) (by simp [ready, setLocal])
    (code.drop 10) (number values.sum) (collectorHeap base values)
    (by intro after collector; simpa only [collected] using finale base after collector)
  have answer : values.sum = ProjectEulerStatements.P37.naive 1000000 := by rw [values_sum, reference_value]
  refine ⟨collectorHeap base values, ?_⟩
  rw [← answer]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) sieved)
  apply BlockReturns.assertion
    (Evaluates.compare (av := number 1) (bv := number 1)
      (middle := base) (afterRight := base) (by simpa only [if_pos primeExample] using exampleRead) Evaluates.constant rfl) rfl
  apply BlockReturns.assertion (by simpa only [example_proper] using exampleProper) rfl
  apply BlockReturns.assertion (not_small_evaluates 2 (heap.length + 1) (by decide) bytes base sieveEnv (by omega) found correct rfl) rfl
  apply BlockReturns.assertion (not_small_evaluates 3 (heap.length + 1) (by decide) bytes base sieveEnv (by omega) found correct rfl) rfl
  apply BlockReturns.assertion (not_small_evaluates 5 (heap.length + 1) (by decide) bytes base sieveEnv (by omega) found correct rfl) rfl
  apply BlockReturns.assertion (not_small_evaluates 7 (heap.length + 1) (by decide) bytes base sieveEnv (by omega) found correct rfl) rfl
  apply BlockContinues.returns (BlockContinues.annAssign (Evaluates.list .nil))
  apply BlockReturns.for_range (values := [number 11, number 1000001, number 2]) (middle := collectorHeap base [])
    (.cons Evaluates.constant (.cons
      (Evaluates.binary (av := number 1000000) (bv := number 1) (value := number 1000001)
        (middle := collectorHeap base []) (afterRight := collectorHeap base []) (finalHeap := collectorHeap base [])
        (Evaluates.lookup (by simp [sieveEnv, limitEnv, setLocal])) Evaluates.constant rfl)
      (.cons Evaluates.constant .nil))) rfl
  exact run

theorem «Equiv(naive_default)» :
    (HeapCalls.scalarFunction program "solve" []).Equiv
      (HeapCalls.scalarFunction P37Ast.module "naive_default" []) := by
  obtain ⟨_, optimized⟩ := result []
  obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p37.naive_default.result []
  have first := optimized.scalar_finishes (values := [])
  have second := reference.scalar_finishes (values := [])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique first]
    exact second
  · intro finished
    rw [finished.unique second]
    exact first

theorem «Supported» :
    (HeapCalls.scalarFunction program "solve" []).Supported ∧
      (HeapCalls.scalarFunction P37Ast.module "naive_default" []).Supported := by
  obtain ⟨_, optimized⟩ := result []
  exact ⟨(optimized.scalar_returns (values := [])).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p37.naive_default.«Supported»⟩

end solve

end solvers.«37»
