import Lean
import ProjectEulerSolutions.P1
import ProjectEulerSolutions.P2
import ProjectEulerSolutions.P3
import ProjectEulerSolutions.P4
import ProjectEulerSolutions.P5
import ProjectEulerSolutions.P6
import ProjectEulerSolutions.P7
import ProjectEulerSolutions.P8
import ProjectEulerSolutions.P9
import ProjectEulerSolutions.P10
import ProjectEulerSolutions.P11
import ProjectEulerSolutions.P12
import ProjectEulerSolutions.P13
import ProjectEulerSolutions.P14
import ProjectEulerSolutions.P15
import ProjectEulerSolutions.P16
import ProjectEulerSolutions.P17
import ProjectEulerSolutions.P18
import ProjectEulerSolutions.P19
import ProjectEulerSolutions.P20
import ProjectEulerSolutions.P21
import ProjectEulerSolutions.P22
import ProjectEulerSolutions.P23
import ProjectEulerSolutions.P24
import ProjectEulerSolutions.P25
import ProjectEulerSolutions.P26
import ProjectEulerSolutions.P27
import ProjectEulerSolutions.P28
import ProjectEulerSolutions.P29
import ProjectEulerSolutions.P30
import ProjectEulerSolutions.P31
import ProjectEulerSolutions.P32
import ProjectEulerSolutions.P33
import ProjectEulerSolutions.P34
import ProjectEulerSolutions.P35
import ProjectEulerSolutions.P36
import ProjectEulerSolutions.P37
import ProjectEulerSolutions.P38
import ProjectEulerSolutions.P39
import ProjectEulerSolutions.P40
import ProjectEulerSolutions.P41
import ProjectEulerSolutions.P42
import ProjectEulerSolutions.P43
import ProjectEulerSolutions.P44
import ProjectEulerSolutions.P45
import ProjectEulerSolutions.P46
import ProjectEulerSolutions.P47
import ProjectEulerSolutions.P48
import ProjectEulerSolutions.P49
import ProjectEulerSolutions.P50
import ProjectEulerSolutions.P51
import ProjectEulerSolutions.P52
import ProjectEulerSolutions.P53
import ProjectEulerSolutions.P54
import ProjectEulerSolutions.P55
import ProjectEulerSolutions.P56
import ProjectEulerSolutions.P57
import ProjectEulerSolutions.P58
import ProjectEulerSolutions.P59
import ProjectEulerSolutions.P60
import ProjectEulerSolutions.P61
import ProjectEulerSolutions.P62
import ProjectEulerSolutions.P63
import ProjectEulerSolutions.P64
import ProjectEulerSolutions.P65
import ProjectEulerSolutions.P66
import ProjectEulerSolutions.P67
import ProjectEulerSolutions.P68
import ProjectEulerSolutions.P69
import ProjectEulerSolutions.P70
import ProjectEulerSolutions.P71
import ProjectEulerSolutions.P72
import ProjectEulerSolutions.P73
import ProjectEulerSolutions.P74
import ProjectEulerSolutions.P75
import ProjectEulerSolutions.P76
import ProjectEulerSolutions.P77
import ProjectEulerSolutions.P78
import ProjectEulerSolutions.P79
import ProjectEulerSolutions.P80
import ProjectEulerSolutions.P81
import ProjectEulerSolutions.P82
import ProjectEulerSolutions.P83
import ProjectEulerSolutions.P84
import ProjectEulerSolutions.P85
import ProjectEulerSolutions.P86
import ProjectEulerSolutions.P87
import ProjectEulerSolutions.P88
import ProjectEulerSolutions.P89
import ProjectEulerSolutions.P90
import ProjectEulerSolutions.P91
import ProjectEulerSolutions.P92
import ProjectEulerSolutions.P93
import ProjectEulerSolutions.P94
import ProjectEulerSolutions.P95
import ProjectEulerSolutions.P96
import ProjectEulerSolutions.P97
import ProjectEulerSolutions.P98
import ProjectEulerSolutions.P99
import ProjectEulerSolutions.P100

open Lean Elab Command Meta

private def isSolveApp (e : Expr) (solveName : Name) : Bool :=
  let e := e.consumeMData
  match e.getAppFn with
  | Expr.const name _ => name == solveName
  | _ => false

private def isSerializeSolve (e : Expr) (solveName : Name) : Bool :=
  let e := e.consumeMData
  match e.getAppFn with
  | Expr.const name _ =>
      let nameStr := name.toString
      if !nameStr.endsWith ".serialize" && nameStr != "serialize" then
        false
      else
        let args := e.getAppArgs
        match args[(args.size - 1)]? with
        | some last => isSolveApp last solveName
        | none => false
  | _ => false

/- Check that `main` prints either `solve ...` or `serialize (solve ...)`
   without caring about the number of arguments to `solve`. -/
private def isPrintSolve (e : Expr) (solveName : Name) : Bool :=
  let e := e.consumeMData
  match e.getAppFn with
  | Expr.const name _ =>
      if name != ``IO.println then
        false
      else
        let args := e.getAppArgs
        match args[(args.size - 1)]? with
        | some last => isSolveApp last solveName || isSerializeSolve last solveName
        | none => false
  | _ => false

syntax (name := checkMainPrintsSolve) "check_main_prints_solve " term " with " term : command

elab_rules : command
  | `(check_main_prints_solve $mainTerm with $solveTerm) => do
      liftTermElabM do
        let mainExpr ← Term.elabTerm mainTerm none
        let solveExpr ← Term.elabTerm solveTerm none
        let mainName ←
          match mainExpr with
          | Expr.const name _ => pure name
          | _ => throwError "main must be a constant name"
        let solveName ←
          match solveExpr with
          | Expr.const name _ => pure name
          | _ => throwError "solve must be a constant name"
        let env ← getEnv
        let some mainDecl := env.find? mainName
          | throwError m!"unknown main definition '{mainName}'"
        let some mainVal := mainDecl.value?
          | throwError m!"main '{mainName}' has no value"
        let mainBody := mainVal
        unless isPrintSolve mainBody solveName do
          throwError
            m!"main '{mainName}' is not of the form 'IO.println (solve ...)'"

def p1solve := ProjectEulerSolutions.P1.solve

def p1equiv : ProjectEulerStatements.P1.naive = p1solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P1.equiv

def p2solve := ProjectEulerSolutions.P2.solve

def p2equiv : ProjectEulerStatements.P2.naive = p2solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P2.equiv

def p3solve := ProjectEulerSolutions.P3.solve

def p3equiv : ProjectEulerStatements.P3.naive = p3solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P3.equiv

def p4solve := ProjectEulerSolutions.P4.solve

def p4equiv : ProjectEulerStatements.P4.naive = p4solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P4.equiv

def p5solve := ProjectEulerSolutions.P5.solve

def p5equiv : ProjectEulerStatements.P5.naive = p5solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P5.equiv

def p6solve := ProjectEulerSolutions.P6.solve

def p6equiv : ProjectEulerStatements.P6.naive = p6solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P6.equiv

def p7solve := ProjectEulerSolutions.P7.solve

def p7equiv : ProjectEulerStatements.P7.naive = p7solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P7.equiv

def p8solve := ProjectEulerSolutions.P8.solve

def p8equiv : ProjectEulerStatements.P8.naive = p8solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P8.equiv

def p9solve := ProjectEulerSolutions.P9.solve

def p9equiv : ProjectEulerStatements.P9.naive = p9solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P9.equiv

def p10solve := ProjectEulerSolutions.P10.solve

def p10equiv : ProjectEulerStatements.P10.naive = p10solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P10.equiv

def p11solve := ProjectEulerSolutions.P11.solve

def p11equiv : ProjectEulerStatements.P11.naive = p11solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P11.equiv

def p12solve := ProjectEulerSolutions.P12.solve

def p12equiv : ProjectEulerStatements.P12.naive = p12solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P12.equiv

def p13solve := ProjectEulerSolutions.P13.solve

def p13equiv : ProjectEulerStatements.P13.naive = p13solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P13.equiv

def p14solve := ProjectEulerSolutions.P14.solve

def p14equiv : ProjectEulerStatements.P14.naive = p14solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P14.equiv

def p15solve := ProjectEulerSolutions.P15.solve

def p15equiv : ProjectEulerStatements.P15.naive = p15solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P15.equiv

def p16solve := ProjectEulerSolutions.P16.solve

def p16equiv : ProjectEulerStatements.P16.naive = p16solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P16.equiv

def p17solve := ProjectEulerSolutions.P17.solve

def p17equiv : ProjectEulerStatements.P17.naive = p17solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P17.equiv

def p18solve := ProjectEulerSolutions.P18.solve

def p18equiv : ProjectEulerStatements.P18.naive = p18solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P18.equiv

def p19solve := ProjectEulerSolutions.P19.solve

def p19equiv : ProjectEulerStatements.P19.naive = p19solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P19.equiv

def p20solve := ProjectEulerSolutions.P20.solve

def p20equiv : ProjectEulerStatements.P20.naive = p20solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P20.equiv

def p21solve := ProjectEulerSolutions.P21.solve

def p21equiv : ProjectEulerStatements.P21.naive = p21solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P21.equiv

def p22solve := ProjectEulerSolutions.P22.solve

def p22equiv : ProjectEulerStatements.P22.naive = p22solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P22.equiv

def p23solve := ProjectEulerSolutions.P23.solve

def p23equiv : ProjectEulerStatements.P23.naive = p23solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P23.equiv

def p24solve := ProjectEulerSolutions.P24.solve

def p24equiv : ProjectEulerStatements.P24.naive = p24solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P24.equiv

def p25solve := ProjectEulerSolutions.P25.solve

def p25equiv : ProjectEulerStatements.P25.naive = p25solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P25.equiv

def p26solve := ProjectEulerSolutions.P26.solve

def p26equiv : ProjectEulerStatements.P26.naive = p26solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P26.equiv

def p27solve := ProjectEulerSolutions.P27.solve

def p27equiv : ProjectEulerStatements.P27.naive = p27solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P27.equiv

def p28solve := ProjectEulerSolutions.P28.solve

def p28equiv : ProjectEulerStatements.P28.naive = p28solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P28.equiv

def p29solve := ProjectEulerSolutions.P29.solve

def p29equiv : ProjectEulerStatements.P29.naive = p29solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P29.equiv

def p30solve := ProjectEulerSolutions.P30.solve

def p30equiv : ProjectEulerStatements.P30.naive = p30solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P30.equiv

def p31solve := ProjectEulerSolutions.P31.solve

def p31equiv : ProjectEulerStatements.P31.naive = p31solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P31.equiv

def p32solve := ProjectEulerSolutions.P32.solve

def p32equiv : ProjectEulerStatements.P32.naive = p32solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P32.equiv

def p33solve := ProjectEulerSolutions.P33.solve

def p33equiv : ProjectEulerStatements.P33.naive = p33solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P33.equiv

def p34solve := ProjectEulerSolutions.P34.solve

def p34equiv : ProjectEulerStatements.P34.naive = p34solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P34.equiv

def p35solve := ProjectEulerSolutions.P35.solve

def p35equiv : ProjectEulerStatements.P35.naive = p35solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P35.equiv

def p36solve := ProjectEulerSolutions.P36.solve

def p36equiv : ProjectEulerStatements.P36.naive = p36solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P36.equiv

def p37solve := ProjectEulerSolutions.P37.solve

def p37equiv : ProjectEulerStatements.P37.naive = p37solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P37.equiv

def p38solve := ProjectEulerSolutions.P38.solve

def p38equiv : ProjectEulerStatements.P38.naive = p38solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P38.equiv

def p39solve := ProjectEulerSolutions.P39.solve

def p39equiv : ProjectEulerStatements.P39.naive = p39solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P39.equiv

def p40solve := ProjectEulerSolutions.P40.solve

def p40equiv : ProjectEulerStatements.P40.naive = p40solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P40.equiv

def p41solve := ProjectEulerSolutions.P41.solve

def p41equiv : ProjectEulerStatements.P41.naive = p41solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P41.equiv

def p42solve := ProjectEulerSolutions.P42.solve

def p42equiv : ProjectEulerStatements.P42.naive = p42solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P42.equiv

def p43solve := ProjectEulerSolutions.P43.solve

def p43equiv : ProjectEulerStatements.P43.naive = p43solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P43.equiv

def p44solve := ProjectEulerSolutions.P44.solve

def p44equiv : ProjectEulerStatements.P44.naive = p44solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P44.equiv

def p45solve := ProjectEulerSolutions.P45.solve

def p45equiv : ProjectEulerStatements.P45.naive = p45solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P45.equiv

def p46solve := ProjectEulerSolutions.P46.solve

def p46equiv : ProjectEulerStatements.P46.naive = p46solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P46.equiv

def p47solve := ProjectEulerSolutions.P47.solve

def p47equiv : ProjectEulerStatements.P47.naive = p47solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P47.equiv

def p48solve := ProjectEulerSolutions.P48.solve

def p48equiv : ProjectEulerStatements.P48.naive = p48solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P48.equiv

def p49solve := ProjectEulerSolutions.P49.solve

def p49equiv : ProjectEulerStatements.P49.naive = p49solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P49.equiv

def p50solve := ProjectEulerSolutions.P50.solve

def p50equiv : ProjectEulerStatements.P50.naive = p50solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P50.equiv

def p51solve := ProjectEulerSolutions.P51.solve

def p51equiv : ProjectEulerStatements.P51.naive = p51solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P51.equiv

def p52solve := ProjectEulerSolutions.P52.solve

def p52equiv : ProjectEulerStatements.P52.naive = p52solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P52.equiv

def p53solve := ProjectEulerSolutions.P53.solve

def p53equiv : ProjectEulerStatements.P53.naive = p53solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P53.equiv

def p54solve := ProjectEulerSolutions.P54.solve

def p54equiv : ProjectEulerStatements.P54.naive = p54solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P54.equiv

def p55solve := ProjectEulerSolutions.P55.solve

def p55equiv : ProjectEulerStatements.P55.naive = p55solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P55.equiv

def p56solve := ProjectEulerSolutions.P56.solve

def p56equiv : ProjectEulerStatements.P56.naive = p56solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P56.equiv

def p57solve := ProjectEulerSolutions.P57.solve

def p57equiv : ProjectEulerStatements.P57.naive = p57solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P57.equiv

def p58solve := ProjectEulerSolutions.P58.solve

def p58equiv : ProjectEulerStatements.P58.naive = p58solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P58.equiv

def p59solve := ProjectEulerSolutions.P59.solve

def p59equiv : ProjectEulerStatements.P59.naive = p59solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P59.equiv

def p60solve := ProjectEulerSolutions.P60.solve

def p60equiv : ProjectEulerStatements.P60.naive = p60solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P60.equiv

def p61solve := ProjectEulerSolutions.P61.solve

def p61equiv : ProjectEulerStatements.P61.naive = p61solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P61.equiv

def p62solve := ProjectEulerSolutions.P62.solve

def p62equiv : ProjectEulerStatements.P62.naive = p62solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P62.equiv

def p63solve := ProjectEulerSolutions.P63.solve

def p63equiv : ProjectEulerStatements.P63.naive = p63solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P63.equiv

def p64solve := ProjectEulerSolutions.P64.solve

def p64equiv : ProjectEulerStatements.P64.naive = p64solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P64.equiv

def p65solve := ProjectEulerSolutions.P65.solve

def p65equiv : ProjectEulerStatements.P65.naive = p65solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P65.equiv

def p66solve := ProjectEulerSolutions.P66.solve

def p66equiv : ProjectEulerStatements.P66.naive = p66solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P66.equiv

def p67solve := ProjectEulerSolutions.P67.solve

def p67equiv : ProjectEulerStatements.P67.naive = p67solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P67.equiv

def p68solve := ProjectEulerSolutions.P68.solve

def p68equiv : ProjectEulerStatements.P68.naive = p68solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P68.equiv

def p69solve := ProjectEulerSolutions.P69.solve

def p69equiv : ProjectEulerStatements.P69.naive = p69solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P69.equiv

def p70solve := ProjectEulerSolutions.P70.solve

def p70equiv : ProjectEulerStatements.P70.naive = p70solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P70.equiv

def p71solve := ProjectEulerSolutions.P71.solve

def p71equiv : ProjectEulerStatements.P71.naive = p71solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P71.equiv

def p72solve := ProjectEulerSolutions.P72.solve

def p72equiv : ProjectEulerStatements.P72.naive = p72solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P72.equiv

def p73solve := ProjectEulerSolutions.P73.solve

def p73equiv : ProjectEulerStatements.P73.naive = p73solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P73.equiv

def p74solve := ProjectEulerSolutions.P74.solve

def p74equiv : ProjectEulerStatements.P74.naive = p74solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P74.equiv

def p75solve := ProjectEulerSolutions.P75.solve

def p75equiv : ProjectEulerStatements.P75.naive = p75solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P75.equiv

def p76solve := ProjectEulerSolutions.P76.solve

def p76equiv : ProjectEulerStatements.P76.naive = p76solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P76.equiv

def p77solve := ProjectEulerSolutions.P77.solve

def p77equiv : ProjectEulerStatements.P77.naive = p77solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P77.equiv

def p78solve := ProjectEulerSolutions.P78.solve

def p78equiv : ProjectEulerStatements.P78.naive = p78solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P78.equiv

def p79solve := ProjectEulerSolutions.P79.solve

def p79equiv : ProjectEulerStatements.P79.naive = p79solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P79.equiv

def p80solve := ProjectEulerSolutions.P80.solve

def p80equiv : ProjectEulerStatements.P80.naive = p80solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P80.equiv

def p81solve := ProjectEulerSolutions.P81.solve

def p81equiv : ProjectEulerStatements.P81.naive = p81solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P81.equiv

def p82solve := ProjectEulerSolutions.P82.solve

def p82equiv : ProjectEulerStatements.P82.naive = p82solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P82.equiv

def p83solve := ProjectEulerSolutions.P83.solve

def p83equiv : ProjectEulerStatements.P83.naive = p83solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P83.equiv

def p84solve := ProjectEulerSolutions.P84.solve

def p84equiv : ProjectEulerStatements.P84.naive = p84solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P84.equiv

def p85solve := ProjectEulerSolutions.P85.solve

def p85equiv : ProjectEulerStatements.P85.naive = p85solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P85.equiv

def p86solve := ProjectEulerSolutions.P86.solve

def p86equiv : ProjectEulerStatements.P86.naive = p86solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P86.equiv

def p87solve := ProjectEulerSolutions.P87.solve

def p87equiv : ProjectEulerStatements.P87.naive = p87solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P87.equiv

def p88solve := ProjectEulerSolutions.P88.solve

def p88equiv : ProjectEulerStatements.P88.naive = p88solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P88.equiv

def p89solve := ProjectEulerSolutions.P89.solve

def p89equiv : ProjectEulerStatements.P89.naive = p89solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P89.equiv

def p90solve := ProjectEulerSolutions.P90.solve

def p90equiv : ProjectEulerStatements.P90.naive = p90solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P90.equiv

def p91solve := ProjectEulerSolutions.P91.solve

def p91equiv : ProjectEulerStatements.P91.naive = p91solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P91.equiv

def p92solve := ProjectEulerSolutions.P92.solve

def p92equiv : ProjectEulerStatements.P92.naive = p92solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P92.equiv

def p93solve := ProjectEulerSolutions.P93.solve

def p93equiv : ProjectEulerStatements.P93.naive = p93solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P93.equiv

def p94solve := ProjectEulerSolutions.P94.solve

def p94equiv : ProjectEulerStatements.P94.naive = p94solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P94.equiv

def p95solve := ProjectEulerSolutions.P95.solve

def p95equiv : ProjectEulerStatements.P95.naive = p95solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P95.equiv

def p96solve := ProjectEulerSolutions.P96.solve

def p96equiv : ProjectEulerStatements.P96.naive = p96solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P96.equiv

def p97solve := ProjectEulerSolutions.P97.solve

def p97equiv : ProjectEulerStatements.P97.naive = p97solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P97.equiv

def p98solve := ProjectEulerSolutions.P98.solve

def p98equiv : ProjectEulerStatements.P98.naive = p98solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P98.equiv

def p99solve := ProjectEulerSolutions.P99.solve

def p99equiv : ProjectEulerStatements.P99.naive = p99solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P99.equiv

def p100solve := ProjectEulerSolutions.P100.solve

def p100equiv : ProjectEulerStatements.P100.naive = p100solve := by
  apply (funext_iff).2
  simpa [funext_iff] using ProjectEulerSolutions.P100.equiv
