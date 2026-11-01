import «.formalsw».solvers.«38.py.ast»
import ProjectEulerSolutions.Lib.PandigitalConcat
import lang.py.FormalSW.HeapCalls

set_option Elab.async false

namespace solvers.«38»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P38 ProjectEulerStatements.ConcatenatedProducts ProjectEulerSolutions.PandigitalConcat
private abbrev program := Solvers.P38Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
variable {globals : Mutable.Globals}

namespace concatenated_product
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "concatenated_product").getD ([], [])).2
private abbrev condition : Expr := .compare (.name "total_len") [(.lt, .name "width")]
private abbrev body : List Stmt :=
  match (code[3]? : Option Stmt) with | some (.whileDo _ statements _) => statements | _ => []
private def values (x count : Nat) : List Mutable.Value :=
  (parts x count).map (fun piece => .scalar (.string piece))
private def collectorHeap (base : Mutable.Heap) (x count : Nat) : Mutable.Heap :=
  base ++ [.list (values x count)]
private def Locals (env : HeapCalls.Env) (address x count width : Nat) : Prop :=
  env "x" = some (number x) ∧ env "n" = some (number count) ∧
    env "total_len" = some (number (numDigits (concatProduct x count))) ∧
    env "parts" = some (.ref address) ∧ env "width" = some (number width)
private def nextEnv (env : HeapCalls.Env) (x count : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "n" (number (count+1 : Nat))) "s"
    (.scalar (.string (toString ((x*(count+1) : Nat) : Int))))) "total_len"
    (number (numDigits (concatProduct x (count+1))))

private theorem iteration (width x count : Nat) (positive : 0 < x)
    (base : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals env base.length x count width) :
    BlockContinues (globals := globals) program env body (collectorHeap base x count)
      (nextEnv env x count) (collectorHeap base x (count+1)) ∧
      Locals (nextEnv env x count) base.length x (count+1) width := by
  let e1 := setLocal env "n" (number (count+1 : Nat))
  let piece := toString ((x*(count+1) : Nat) : Int)
  let e2 := setLocal e1 "s" (.scalar (.string piece))
  have first : BlockContinues (globals := globals) program env [body[0]'(by decide)]
      (collectorHeap base x count) e1 (collectorHeap base x count) :=
    BlockContinues.augAssign locals.2.1 Evaluates.constant
      (by simp [inplace, Mutable.binary, scalarBinary, number])
  have second : BlockContinues (globals := globals) program e1 [body[1]'(by decide)]
      (collectorHeap base x count) e2 (collectorHeap base x count) := by
    apply BlockContinues.assign
    apply Evaluates.builtin (.cons
      (Evaluates.binary (av := number x) (bv := number (count+1 : Nat))
        (value := number (x*(count+1) : Nat)) (middle := collectorHeap base x count)
        (afterRight := collectorHeap base x count) (finalHeap := collectorHeap base x count)
        (Evaluates.lookup (by simpa [e1, setLocal] using locals.1)) (Evaluates.lookup rfl)
        (by simp [Mutable.binary, scalarBinary, number])) .nil)
    rfl
  have third : BlockContinues (globals := globals) program e2 [body[2]'(by decide)]
      (collectorHeap base x count) e2 (collectorHeap base x (count+1)) := by
    apply BlockContinues.discard
    have appended := Evaluates.append (globals := globals) (module := program) (env := e2)
      (base := .name "parts") (expression := .name "s") (value := .scalar (.string piece))
      (heap := collectorHeap base x count)
      (Evaluates.lookup (by simpa [e2, e1, setLocal] using locals.2.2.2.1))
      (by simp [collectorHeap] : (collectorHeap base x count)[base.length]? = some (.list (values x count)))
      (Evaluates.lookup rfl)
      (by simp [collectorHeap] : (collectorHeap base x count)[base.length]? = some (.list (values x count)))
    simpa [collectorHeap, values, parts_succ, piece] using appended
  have fourth : BlockContinues (globals := globals) program e2 [body[3]'(by decide)]
      (collectorHeap base x (count+1)) (nextEnv env x count) (collectorHeap base x (count+1)) := by
    have lengthResult : Evaluates (globals := globals) program e2 (.builtinCall .len [.name "s"])
        (collectorHeap base x (count+1)) (number (numDigits (x*(count+1)))) (collectorHeap base x (count+1)) := by
      apply Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil)
      have equal := ProjectEulerStatements.DecimalDigits.int_string_length (x*(count+1))
        (Nat.mul_pos positive (by omega))
      simpa [primitive, Mutable.length, number, numDigits, piece, equal]

    apply BlockContinues.augAssign
      (left := number (numDigits (concatProduct x count)))
      (right := number (numDigits (x*(count+1))))
      (value := number (numDigits (concatProduct x (count+1))))
      (middle := collectorHeap base x (count+1)) (finalHeap := collectorHeap base x (count+1))
      (by simpa [e2, e1, setLocal] using locals.2.2.1) lengthResult
    simp [inplace, Mutable.binary, scalarBinary, number, concatProduct_length_succ]
  refine ⟨first.append (second.append (third.append fourth)), ?_⟩
  rcases locals with ⟨getX, getN, getLength, getParts, getWidth⟩
  exact ⟨by simpa [nextEnv, setLocal] using getX, by simp [nextEnv, setLocal],
    by simp [nextEnv, setLocal], by simpa [nextEnv, setLocal] using getParts,
    by simpa [nextEnv, setLocal] using getWidth⟩

private theorem loop (width x count remaining : Nat) (positive : 0 < x)
    (endpoint : count+remaining = stop width x) (base : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env base.length x count width) :
    ∃ finalEnv, BlockContinues (globals := globals) program env [.whileDo condition body []]
      (collectorHeap base x count) finalEnv (collectorHeap base x (stop width x)) ∧
      Locals finalEnv base.length x (stop width x) width := by
  induction remaining generalizing count env with
  | zero =>
      have same : count = stop width x := by omega
      have stopped : ¬ numDigits (concatProduct x count) < width := by
        rw [same]
        have reached := stop_spec width x positive
        omega
      have tested : Evaluates (globals := globals) program env condition (collectorHeap base x count)
          (.scalar (.bool false)) (collectorHeap base x count) :=
        Evaluates.compare (Evaluates.lookup locals.2.2.1) (Evaluates.lookup locals.2.2.2.2)
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, stopped])
      refine ⟨env, ?_, by simpa only [same] using locals⟩
      rw [← same]
      exact BlockContinues.while_loop tested rfl .nil
  | succ remaining ih =>
      have before : count < stop width x := by omega
      have continues := before_stop width x count positive before
      have tested : Evaluates (globals := globals) program env condition (collectorHeap base x count)
          (.scalar (.bool true)) (collectorHeap base x count) :=
        Evaluates.compare (Evaluates.lookup locals.2.2.1) (Evaluates.lookup locals.2.2.2.2)
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, continues])
      obtain ⟨iterated, nextLocals⟩ := iteration (globals := globals) width x count positive base env locals
      obtain ⟨finalEnv, rest, finalLocals⟩ := ih (count+1) (by omega) (nextEnv env x count) nextLocals
      refine ⟨finalEnv, ?_, finalLocals⟩
      apply BlockContinues.while_loop (flag := true) tested rfl
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using iterated.append rest

/-- The list-building loop reaches the requested width and returns the exact positional concatenation. -/
theorem result (width x : Nat) (nonempty : 0 < width) (positive : 0 < x) (heap : Mutable.Heap) :
    ReturnsValue (globals := globals) program "concatenated_product" [number x, number width] heap
      (.ref (heap.length+1))
      (heap ++ [.list (values x (stop width x)),
        .tuple [.scalar (.string (toString (concatProduct x (stop width x) : Int))), number (stop width x)]]) := by
  let env := functionEnv ((findFunctionIn program.body "concatenated_product").getD ([], [])).1
    ((findFunctionIn program.body "concatenated_product").getD ([], [])).2 [number x, number width] globals
  let e1 := setLocal env "parts" (.ref heap.length)
  let e2 := setLocal e1 "n" (number 0)
  let ready := setLocal e2 "total_len" (number 0)
  have initialLocals : Locals ready heap.length x 0 width := by
    refine ⟨?_, rfl, rfl, rfl, ?_⟩
    all_goals cases globals <;> rfl
  obtain ⟨finalEnv, searched, finalLocals⟩ :=
    loop (globals := globals) width x 0 (stop width x) positive (by omega) heap ready initialLocals
  have joined : Evaluates (globals := globals) program finalEnv
      (.call (.attribute (.constant (.string "")) "join") [.name "parts"])
      (collectorHeap heap x (stop width x))
      (.scalar (.string (toString (concatProduct x (stop width x) : Int))))
      (collectorHeap heap x (stop width x)) := by
    apply Evaluates.join Evaluates.constant (.cons (Evaluates.lookup finalLocals.2.2.2.1) .nil)
    · have operation := primitive_join_list (collectorHeap heap x (stop width x)) heap.length ""
        (parts x (stop width x)) (by simp [collectorHeap, values])
      simpa only [intercalate_parts, text_decimal x (stop width x) positive
        (by have := (stop_bounds width x nonempty positive).1; omega)] using operation
    · intro name impossible
      cases impossible
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .ref heap.length) (middle := heap ++ [.list []]) (Evaluates.list .nil)
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  apply BlockContinues.returns searched
  apply BlockReturns.return_value
  have returned := Evaluates.tuple (globals := globals)
    (.cons joined (.cons (Evaluates.lookup finalLocals.2.1) .nil)) (by simp)
  simpa [collectorHeap, List.append_assoc] using returned

end concatenated_product
end solvers.«38»
