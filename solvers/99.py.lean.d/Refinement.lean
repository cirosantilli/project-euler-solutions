import solvers.«99.py.lean.d».Scaled

namespace solvers.«99».compare_powers
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PowerIntervals
set_option Elab.async false

abbrev leftNames : List String := ["lo", "hi", "shift"]
abbrev rightNames : List String := ["other_lo", "other_hi", "other_shift"]
abbrev setup : List Stmt :=
  [.assign [.tuple (leftNames.map Expr.name)]
    (.call (.name "power_bounds") [.name "left_base", .name "left_exp", .name "precision"]),
   .assign [.tuple (rightNames.map Expr.name)]
    (.call (.name "power_bounds") [.name "right_base", .name "right_exp", .name "precision"])]
abbrev greaterCall : Expr :=
  .call (.name "compare_scaled") [.name "lo", .name "shift", .name "other_hi", .name "other_shift"]
abbrev lessCall : Expr :=
  .call (.name "compare_scaled") [.name "hi", .name "shift", .name "other_lo", .name "other_shift"]
abbrev exactCall : Expr :=
  .call (.name "compare_scaled") [.name "lo", .name "shift", .name "other_lo", .name "other_shift"]
abbrev exactTest : Expr :=
  .boolOp .and [.compare (.name "lo") [(.eq, .name "hi")],
    .compare (.name "other_lo") [(.eq, .name "other_hi")]]
abbrev tail : List Stmt :=
  [.ifThen (.compare greaterCall [(.gt, .constant (.int 0))]) [.return (.name "direction")] [],
   .ifThen (.compare lessCall [(.lt, .constant (.int 0))]) [.return (.unaryOp .uSub (.name "direction"))] [],
   .ifThen exactTest [.return (.binOp (.name "direction") .mult exactCall)] [],
   .augAssign (.name "precision") .mult (.constant (.int 2))]
abbrev body : List Stmt := setup ++ tail
abbrev loop : Stmt := .whileDo (.constant (.bool true)) body []

theorem source_loop :
    (((findFunctionIn program.body "compare_powers").getD ([], [])).2).getLast? = some loop := rfl

structure Locals (env : HeapCalls.Env) (a m b n precision : Nat) (direction : Int) : Prop where
  getA : env "left_base" = some (number a)
  getM : env "left_exp" = some (number m)
  getB : env "right_base" = some (number b)
  getN : env "right_exp" = some (number n)
  getPrecision : env "precision" = some (number precision)
  getDirection : env "direction" = some (number direction)

structure Ready (env : HeapCalls.Env) (a m b n precision : Nat) (direction : Int)
    extends Locals env a m b n precision direction where
  getLo : env "lo" = some (number (power a m precision).lower)
  getHi : env "hi" = some (number (power a m precision).upper)
  getShift : env "shift" = some (number (power a m precision).shift)
  getOtherLo : env "other_lo" = some (number (power b n precision).lower)
  getOtherHi : env "other_hi" = some (number (power b n precision).upper)
  getOtherShift : env "other_shift" = some (number (power b n precision).shift)

theorem setup_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (a m b n precision : Nat) (direction : Int) (locals : Locals env a m b n precision direction) :
    ∃ nextEnv extra, BlockContinues program env setup heap nextEnv (heap ++ extra) globals ∧
      Ready nextEnv a m b n precision direction := by
  obtain ⟨leftExtra, leftRan⟩ := power_bounds.result a m precision heap globals
  let afterLeft := (heap ++ leftExtra) ++ [boundsObject (power a m precision)]
  let leftEnv := assignNames env leftNames (power_bounds.values (power a m precision))
  obtain ⟨rightExtra, rightRan⟩ := power_bounds.result b n precision afterLeft globals
  let rightEnv := assignNames leftEnv rightNames (power_bounds.values (power b n precision))
  refine ⟨rightEnv, leftExtra ++ [boundsObject (power a m precision)] ++ rightExtra ++
    [boundsObject (power b n precision)], ?_, ?_⟩
  · apply BlockContinues.append (body := setup.take 1) (rest := setup.drop 1)
      (middleEnv := leftEnv) (middleHeap := afterLeft)
    · apply BlockContinues.unpack (items := power_bounds.values (power a m precision)) rfl
      · exact Evaluates.call (.cons (Evaluates.lookup locals.getA) (.cons (Evaluates.lookup locals.getM)
          (.cons (Evaluates.lookup locals.getPrecision) .nil))) leftRan
      · simp [Mutable.sequence, boundsObject, power_bounds.values, afterLeft]
      · rfl
    · have rightCall := Evaluates.call (env := leftEnv)
        (.cons (Evaluates.lookup (name := "right_base") (by simpa [leftEnv, leftNames, assignNames, setLocal] using locals.getB))
          (.cons (Evaluates.lookup (name := "right_exp") (by simpa [leftEnv, leftNames, assignNames, setLocal] using locals.getN))
            (.cons (Evaluates.lookup (name := "precision") (by simpa [leftEnv, leftNames, assignNames, setLocal] using locals.getPrecision)) .nil))) rightRan
      simpa only [afterLeft, List.append_assoc] using
        (BlockContinues.unpack (names := rightNames) (expressions := rightNames.map Expr.name)
          (items := power_bounds.values (power b n precision))
          (finalHeap := (afterLeft ++ rightExtra) ++ [boundsObject (power b n precision)]) rfl rightCall
          (by simp [Mutable.sequence, boundsObject, power_bounds.values]) rfl)
  · have preserved : Locals rightEnv a m b n precision direction := by
      rcases locals with ⟨ha, hm, hb, hn, hp, hd⟩
      constructor <;> simp_all [rightEnv, leftEnv, leftNames, rightNames, assignNames, setLocal]
    refine ⟨preserved, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      simp [rightEnv, leftEnv, leftNames, rightNames, assignNames, setLocal, power_bounds.values]

theorem greater_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (a m b n precision : Nat) (direction : Int) (ready : Ready env a m b n precision direction) :
    Evaluates program env greaterCall heap
      (number (scaledComparison (power a m precision).lower (power a m precision).shift
        (power b n precision).upper (power b n precision).shift)) heap globals := by
  exact Evaluates.call (.cons (Evaluates.lookup ready.getLo) (.cons (Evaluates.lookup ready.getShift)
    (.cons (Evaluates.lookup ready.getOtherHi) (.cons (Evaluates.lookup ready.getOtherShift) .nil))))
    (compare_scaled.result _ _ _ _ heap globals)

theorem less_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (a m b n precision : Nat) (direction : Int) (ready : Ready env a m b n precision direction) :
    Evaluates program env lessCall heap
      (number (scaledComparison (power a m precision).upper (power a m precision).shift
        (power b n precision).lower (power b n precision).shift)) heap globals := by
  exact Evaluates.call (.cons (Evaluates.lookup ready.getHi) (.cons (Evaluates.lookup ready.getShift)
    (.cons (Evaluates.lookup ready.getOtherLo) (.cons (Evaluates.lookup ready.getOtherShift) .nil))))
    (compare_scaled.result _ _ _ _ heap globals)

theorem exact_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (a m b n precision : Nat) (direction : Int) (ready : Ready env a m b n precision direction) :
    Evaluates program env exactCall heap
      (number (scaledComparison (power a m precision).lower (power a m precision).shift
        (power b n precision).lower (power b n precision).shift)) heap globals := by
  exact Evaluates.call (.cons (Evaluates.lookup ready.getLo) (.cons (Evaluates.lookup ready.getShift)
    (.cons (Evaluates.lookup ready.getOtherLo) (.cons (Evaluates.lookup ready.getOtherShift) .nil))))
    (compare_scaled.result _ _ _ _ heap globals)

theorem exact_test_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (a m b n precision : Nat) (direction : Int) (ready : Ready env a m b n precision direction) :
    Evaluates program env exactTest heap (.scalar (.bool (decide (
      (power a m precision).lower = (power a m precision).upper ∧
      (power b n precision).lower = (power b n precision).upper)))) heap globals := by
  by_cases left : (power a m precision).lower = (power a m precision).upper
  · have first : Evaluates program env (.compare (.name "lo") [(.eq, .name "hi")])
        heap (.scalar (.bool true)) heap globals :=
      Evaluates.compare (Evaluates.lookup ready.getLo) (Evaluates.lookup ready.getHi)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, left])
    apply Evaluates.boolean_next (flag := true) first rfl rfl
    apply Evaluates.boolean_last
    · apply Evaluates.compare (Evaluates.lookup ready.getOtherLo) (Evaluates.lookup ready.getOtherHi)
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, left]
    · rfl
  · have first : Evaluates program env (.compare (.name "lo") [(.eq, .name "hi")])
        heap (.scalar (.bool false)) heap globals :=
      Evaluates.compare (Evaluates.lookup ready.getLo) (Evaluates.lookup ready.getHi)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, left])
    simpa only [left, false_and, decide_false] using
      (Evaluates.boolean_stop (op := BoolOp.and) (rest := [.compare (.name "other_lo") [(.eq, .name "other_hi")]])
        (flag := false) first rfl rfl)

theorem loop_result (a m b n : Nat) (positiveA : 0 < a) (positiveB : 0 < b)
    (precision : Nat) (positivePrecision : 0 < precision) (direction : Int)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (locals : Locals env a m b n precision direction) :
    ∃ extra, BlockReturns program env [loop] heap
      (number (direction * refine a m b n positiveA positiveB precision positivePrecision)) (heap ++ extra) globals := by
  obtain ⟨readyEnv, garbage, initialized, ready⟩ := setup_result env heap globals a m b n precision direction locals
  let after := heap ++ garbage
  have firstTest := Evaluates.compare (greater_result readyEnv after globals a m b n precision direction ready)
    Evaluates.constant
    (show HeapCalls.compare .gt
      (number (scaledComparison (power a m precision).lower (power a m precision).shift
        (power b n precision).upper (power b n precision).shift)) (.scalar (.int 0)) after =
      some (.ok (.scalar (.bool (decide (0 < scaledComparison (power a m precision).lower (power a m precision).shift
        (power b n precision).upper (power b n precision).shift)))), after) by
      simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases greater : 0 < scaledComparison (power a m precision).lower (power a m precision).shift
      (power b n precision).upper (power b n precision).shift
  · refine ⟨garbage, ?_⟩
    rw [refine, if_pos greater, Int.mul_one]
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    apply initialized.returns
    apply BlockReturns.branch (flag := true) (by simpa [greater] using firstTest) rfl
    exact BlockReturns.return_value (Evaluates.lookup ready.getDirection)
  · have secondTest := Evaluates.compare (less_result readyEnv after globals a m b n precision direction ready)
      Evaluates.constant
      (show HeapCalls.compare .lt
        (number (scaledComparison (power a m precision).upper (power a m precision).shift
          (power b n precision).lower (power b n precision).shift)) (.scalar (.int 0)) after =
        some (.ok (.scalar (.bool (decide (scaledComparison (power a m precision).upper (power a m precision).shift
          (power b n precision).lower (power b n precision).shift < 0)))), after) by
        simp [HeapCalls.compare, scalarCompare, evalCmpOp])
    by_cases less : scaledComparison (power a m precision).upper (power a m precision).shift
        (power b n precision).lower (power b n precision).shift < 0
    · refine ⟨garbage, ?_⟩
      rw [refine, if_neg greater, if_pos less]
      apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
      apply initialized.returns
      apply BlockReturns.branch (flag := false) (by simpa [greater] using firstTest) rfl
      apply BlockReturns.branch (flag := true) (by simpa [less] using secondTest) rfl
      apply BlockReturns.return_value
      exact Evaluates.unary (Evaluates.lookup ready.getDirection)
        (by simp [HeapCalls.unary, scalarExpr, scalarResult, number, after])
    · by_cases exactBounds : (power a m precision).lower = (power a m precision).upper ∧
          (power b n precision).lower = (power b n precision).upper
      · refine ⟨garbage, ?_⟩
        rw [refine, if_neg greater, if_neg less, dif_pos exactBounds]
        apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
        apply initialized.returns
        apply BlockReturns.branch (flag := false) (by simpa [greater] using firstTest) rfl
        apply BlockReturns.branch (flag := false) (by simpa [less] using secondTest) rfl
        apply BlockReturns.branch (flag := true)
          (by simpa [exactBounds] using exact_test_result readyEnv after globals a m b n precision direction ready) rfl
        apply BlockReturns.return_value
        exact Evaluates.binary (Evaluates.lookup ready.getDirection)
          (exact_result readyEnv after globals a m b n precision direction ready)
          (by simp [Mutable.binary, scalarBinary, number, after])
      · let nextEnv := setLocal readyEnv "precision" (number (2 * precision : Nat))
        have nextLocals : Locals nextEnv a m b n (2 * precision) direction := by
          rcases ready.toLocals with ⟨ha, hm, hb, hn, hp, hd⟩
          constructor <;> simp_all [nextEnv, setLocal]
        obtain ⟨extra, rest⟩ := loop_result a m b n positiveA positiveB (2 * precision) (by omega)
          direction nextEnv after globals nextLocals
        refine ⟨garbage ++ extra, ?_⟩
        rw [refine, if_neg greater, if_neg less, dif_neg exactBounds]
        apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
        apply initialized.returns
        apply BlockReturns.branch (flag := false) (by simpa [greater] using firstTest) rfl
        apply BlockReturns.branch (flag := false) (by simpa [less] using secondTest) rfl
        apply BlockReturns.branch (flag := false)
          (by simpa [exactBounds] using exact_test_result readyEnv after globals a m b n precision direction ready) rfl
        apply BlockReturns.augAssign ready.getPrecision Evaluates.constant
          (show inplace .mult (number precision) (number 2) after =
            some (.ok (number (2 * precision : Nat)), after) by
            simp [inplace, Mutable.binary, scalarBinary, number, mul_comm])
        simpa only [after, nextEnv, List.append_assoc] using rest
termination_by max (a ^ m).size (b ^ n).size - precision
decreasing_by
  have below := inexact_precision_lt a m b n precision positiveA positiveB exactBounds
  omega

theorem loop_correct (a m b n : Nat) (positiveA : 0 < a) (positiveB : 0 < b)
    (precision : Nat) (positivePrecision : 0 < precision) (direction : Int)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (locals : Locals env a m b n precision direction) :
    ∃ extra, BlockReturns program env [loop] heap
      (number (direction * comparison (a ^ m) (b ^ n))) (heap ++ extra) globals := by
  simpa only [refine_eq] using
    loop_result a m b n positiveA positiveB precision positivePrecision direction env heap globals locals

end solvers.«99».compare_powers
