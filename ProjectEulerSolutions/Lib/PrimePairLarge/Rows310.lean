import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10133_small : List Nat := []
theorem row10133_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10133 row10133_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10133_block_005 : List Nat := []
def plan10133_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), .first, .second, .second, (.firstFactor 13), .first, (.secondFactor 7), .first, .second, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row10133_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10133 row10133_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10133_block_005
  · decide
  · rfl

def row10133_block_006 : List Nat := [10433, 10631, 10799, 10937, 11423, 11801, 12401]
def plan10133_block_006 : List Budget.Choice := [(.secondFactor 19), .first, .first, .first, .possible, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 37), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 23), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 29), (.secondFactor 7), .first, (.secondFactor 37), (.firstFactor 19), (.firstFactor 29), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10133_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10133 row10133_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10133_block_006
  · decide
  · rfl

def row10133_block_007 : List Nat := [12653, 12689, 13907, 13997]
def plan10133_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .second, .possible, (.firstFactor 17), (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 31), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 17), (.secondFactor 23), .first, .possible, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .possible, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row10133_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10133 row10133_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10133_block_007
  · decide
  · rfl

def row10133_block_008 : List Nat := [14543, 15497]
def plan10133_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 23), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, .second, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .second, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 23), (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 13), .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10133_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10133 row10133_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10133_block_008
  · decide
  · rfl

def row10133_block_009 : List Nat := []
theorem row10133_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10133 row10133_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10133_block_010 : List Nat := []
theorem row10133_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10133 row10133_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10133_block_011 : List Nat := []
theorem row10133_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10133 row10133_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10133_block_012 : List Nat := []
theorem row10133_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10133 row10133_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10133_block_013 : List Nat := []
theorem row10133_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10133 row10133_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10133_large_left_left : List Nat := row10133_small ++ row10133_block_005
theorem row10133_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10133 row10133_large_left_left :=
  Budget.RowUpperBound.branch row10133_small_covers row10133_block_005_covers

def row10133_large_left_right_right : List Nat := row10133_block_007 ++ row10133_block_008
theorem row10133_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10133 row10133_large_left_right_right :=
  Budget.RowUpperBound.branch row10133_block_007_covers row10133_block_008_covers

def row10133_large_left_right : List Nat := row10133_block_006 ++ row10133_large_left_right_right
theorem row10133_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10133 row10133_large_left_right :=
  Budget.RowUpperBound.branch row10133_block_006_covers row10133_large_left_right_right_covers

def row10133_large_left : List Nat := row10133_large_left_left ++ row10133_large_left_right
theorem row10133_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10133 row10133_large_left :=
  Budget.RowUpperBound.branch row10133_large_left_left_covers row10133_large_left_right_covers

def row10133_large_right_left : List Nat := row10133_block_009 ++ row10133_block_010
theorem row10133_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10133 row10133_large_right_left :=
  Budget.RowUpperBound.branch row10133_block_009_covers row10133_block_010_covers

def row10133_large_right_right_right : List Nat := row10133_block_012 ++ row10133_block_013
theorem row10133_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10133 row10133_large_right_right_right :=
  Budget.RowUpperBound.branch row10133_block_012_covers row10133_block_013_covers

def row10133_large_right_right : List Nat := row10133_block_011 ++ row10133_large_right_right_right
theorem row10133_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10133 row10133_large_right_right :=
  Budget.RowUpperBound.branch row10133_block_011_covers row10133_large_right_right_right_covers

def row10133_large_right : List Nat := row10133_large_right_left ++ row10133_large_right_right
theorem row10133_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10133 row10133_large_right :=
  Budget.RowUpperBound.branch row10133_large_right_left_covers row10133_large_right_right_covers

def row10133_large : List Nat := row10133_large_left ++ row10133_large_right
theorem row10133_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10133 row10133_large :=
  Budget.RowUpperBound.branch row10133_large_left_covers row10133_large_right_covers

def row10133 : List Nat := row10133_large
theorem row10133_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10133 row10133 := row10133_large_covers

def row10139_small : List Nat := []
theorem row10139_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10139 row10139_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10139_block_005 : List Nat := []
def plan10139_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 31), (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row10139_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10139 row10139_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10139_block_005
  · decide
  · rfl

def row10139_block_006 : List Nat := [11159, 11273, 11321, 11447, 11813, 12041]
def plan10139_block_006 : List Budget.Choice := [(.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 13), (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .second, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row10139_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10139 row10139_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10139_block_006
  · decide
  · rfl

def row10139_block_007 : List Nat := [12923, 13451, 13679, 14177]
def plan10139_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), (.secondFactor 13), .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 31), .first, .possible, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 37), .first, .possible, (.firstFactor 29), .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10139_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10139 row10139_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10139_block_007
  · decide
  · rfl

def row10139_block_008 : List Nat := [14771, 15269, 15359]
def plan10139_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 23), .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 37), (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), (.secondFactor 13), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 37), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10139_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10139 row10139_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10139_block_008
  · decide
  · rfl

def row10139_block_009 : List Nat := []
theorem row10139_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10139 row10139_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10139_block_010 : List Nat := []
theorem row10139_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10139 row10139_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10139_block_011 : List Nat := []
theorem row10139_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10139 row10139_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10139_block_012 : List Nat := []
theorem row10139_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10139 row10139_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10139_block_013 : List Nat := []
theorem row10139_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10139 row10139_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10139_large_left_left : List Nat := row10139_small ++ row10139_block_005
theorem row10139_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10139 row10139_large_left_left :=
  Budget.RowUpperBound.branch row10139_small_covers row10139_block_005_covers

def row10139_large_left_right_right : List Nat := row10139_block_007 ++ row10139_block_008
theorem row10139_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10139 row10139_large_left_right_right :=
  Budget.RowUpperBound.branch row10139_block_007_covers row10139_block_008_covers

def row10139_large_left_right : List Nat := row10139_block_006 ++ row10139_large_left_right_right
theorem row10139_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10139 row10139_large_left_right :=
  Budget.RowUpperBound.branch row10139_block_006_covers row10139_large_left_right_right_covers

def row10139_large_left : List Nat := row10139_large_left_left ++ row10139_large_left_right
theorem row10139_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10139 row10139_large_left :=
  Budget.RowUpperBound.branch row10139_large_left_left_covers row10139_large_left_right_covers

def row10139_large_right_left : List Nat := row10139_block_009 ++ row10139_block_010
theorem row10139_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10139 row10139_large_right_left :=
  Budget.RowUpperBound.branch row10139_block_009_covers row10139_block_010_covers

def row10139_large_right_right_right : List Nat := row10139_block_012 ++ row10139_block_013
theorem row10139_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10139 row10139_large_right_right_right :=
  Budget.RowUpperBound.branch row10139_block_012_covers row10139_block_013_covers

def row10139_large_right_right : List Nat := row10139_block_011 ++ row10139_large_right_right_right
theorem row10139_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10139 row10139_large_right_right :=
  Budget.RowUpperBound.branch row10139_block_011_covers row10139_large_right_right_right_covers

def row10139_large_right : List Nat := row10139_large_right_left ++ row10139_large_right_right
theorem row10139_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10139 row10139_large_right :=
  Budget.RowUpperBound.branch row10139_large_right_left_covers row10139_large_right_right_covers

def row10139_large : List Nat := row10139_large_left ++ row10139_large_right
theorem row10139_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10139 row10139_large :=
  Budget.RowUpperBound.branch row10139_large_left_covers row10139_large_right_covers

def row10139 : List Nat := row10139_large
theorem row10139_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10139 row10139 := row10139_large_covers

def row10141_small : List Nat := []
theorem row10141_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10141 row10141_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10141_block_005 : List Nat := [10159, 10243]
def plan10141_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row10141_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10141 row10141_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10141_block_005
  · decide
  · rfl

def row10141_block_006 : List Nat := [10711, 11113, 11821, 11887]
def plan10141_block_006 : List Budget.Choice := [.first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .possible, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 29), (.firstFactor 37), .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 11), .first, (.secondFactor 37), .first, (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 19), (.secondFactor 7), .second, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10141_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10141 row10141_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10141_block_006
  · decide
  · rfl

def row10141_block_007 : List Nat := [12763, 13249]
def plan10141_block_007 : List Budget.Choice := [(.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 31), .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 29), (.firstFactor 17), .second, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 19), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 37), .second, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10141_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10141 row10141_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10141_block_007
  · decide
  · rfl

def row10141_block_008 : List Nat := [14557, 14737, 14887, 15199, 15373, 15601]
def plan10141_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .possible, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, .possible, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .possible, (.secondFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .possible, (.secondFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, (.firstFactor 29), .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10141_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10141 row10141_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10141_block_008
  · decide
  · rfl

def row10141_block_009 : List Nat := []
theorem row10141_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10141 row10141_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10141_block_010 : List Nat := []
theorem row10141_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10141 row10141_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10141_block_011 : List Nat := []
theorem row10141_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10141 row10141_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10141_block_012 : List Nat := []
theorem row10141_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10141 row10141_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10141_block_013 : List Nat := []
theorem row10141_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10141 row10141_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10141_large_left_left : List Nat := row10141_small ++ row10141_block_005
theorem row10141_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10141 row10141_large_left_left :=
  Budget.RowUpperBound.branch row10141_small_covers row10141_block_005_covers

def row10141_large_left_right_right : List Nat := row10141_block_007 ++ row10141_block_008
theorem row10141_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10141 row10141_large_left_right_right :=
  Budget.RowUpperBound.branch row10141_block_007_covers row10141_block_008_covers

def row10141_large_left_right : List Nat := row10141_block_006 ++ row10141_large_left_right_right
theorem row10141_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10141 row10141_large_left_right :=
  Budget.RowUpperBound.branch row10141_block_006_covers row10141_large_left_right_right_covers

def row10141_large_left : List Nat := row10141_large_left_left ++ row10141_large_left_right
theorem row10141_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10141 row10141_large_left :=
  Budget.RowUpperBound.branch row10141_large_left_left_covers row10141_large_left_right_covers

def row10141_large_right_left : List Nat := row10141_block_009 ++ row10141_block_010
theorem row10141_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10141 row10141_large_right_left :=
  Budget.RowUpperBound.branch row10141_block_009_covers row10141_block_010_covers

def row10141_large_right_right_right : List Nat := row10141_block_012 ++ row10141_block_013
theorem row10141_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10141 row10141_large_right_right_right :=
  Budget.RowUpperBound.branch row10141_block_012_covers row10141_block_013_covers

def row10141_large_right_right : List Nat := row10141_block_011 ++ row10141_large_right_right_right
theorem row10141_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10141 row10141_large_right_right :=
  Budget.RowUpperBound.branch row10141_block_011_covers row10141_large_right_right_right_covers

def row10141_large_right : List Nat := row10141_large_right_left ++ row10141_large_right_right
theorem row10141_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10141 row10141_large_right :=
  Budget.RowUpperBound.branch row10141_large_right_left_covers row10141_large_right_right_covers

def row10141_large : List Nat := row10141_large_left ++ row10141_large_right
theorem row10141_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10141 row10141_large :=
  Budget.RowUpperBound.branch row10141_large_left_covers row10141_large_right_covers

def row10141 : List Nat := row10141_large
theorem row10141_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10141 row10141 := row10141_large_covers

def row10151_small : List Nat := []
theorem row10151_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10151 row10151_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10151_block_005 : List Nat := []
def plan10151_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 37), (.secondFactor 17), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row10151_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10151 row10151_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10151_block_005
  · decide
  · rfl

def row10151_block_006 : List Nat := [10601, 11159]
def plan10151_block_006 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.secondFactor 29), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 23), .first, .possible, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .second, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 37), .first, .first, .second, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 31), (.secondFactor 19), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 19), .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .second, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10151_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10151 row10151_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10151_block_006
  · decide
  · rfl

def row10151_block_007 : List Nat := [12503, 13109, 13217, 13499, 13859, 14447]
def plan10151_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 19), .first, .first, .first, .second, .first, (.firstFactor 29), (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 31), .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .second, .possible, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 13), (.firstFactor 37), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 23), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 23), .possible, .first, (.secondFactor 37), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 29), .first, .second, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row10151_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10151 row10151_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10151_block_007
  · decide
  · rfl

def row10151_block_008 : List Nat := [14621, 14657, 14951, 15791]
def plan10151_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, .second, .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .possible, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 29), .second, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .possible, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 37), .second, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, (.firstFactor 29), (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10151_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10151 row10151_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10151_block_008
  · decide
  · rfl

def row10151_block_009 : List Nat := []
theorem row10151_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10151 row10151_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10151_block_010 : List Nat := []
theorem row10151_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10151 row10151_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10151_block_011 : List Nat := []
theorem row10151_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10151 row10151_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10151_block_012 : List Nat := []
theorem row10151_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10151 row10151_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10151_block_013 : List Nat := []
theorem row10151_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10151 row10151_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10151_large_left_left : List Nat := row10151_small ++ row10151_block_005
theorem row10151_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10151 row10151_large_left_left :=
  Budget.RowUpperBound.branch row10151_small_covers row10151_block_005_covers

def row10151_large_left_right_right : List Nat := row10151_block_007 ++ row10151_block_008
theorem row10151_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10151 row10151_large_left_right_right :=
  Budget.RowUpperBound.branch row10151_block_007_covers row10151_block_008_covers

def row10151_large_left_right : List Nat := row10151_block_006 ++ row10151_large_left_right_right
theorem row10151_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10151 row10151_large_left_right :=
  Budget.RowUpperBound.branch row10151_block_006_covers row10151_large_left_right_right_covers

def row10151_large_left : List Nat := row10151_large_left_left ++ row10151_large_left_right
theorem row10151_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10151 row10151_large_left :=
  Budget.RowUpperBound.branch row10151_large_left_left_covers row10151_large_left_right_covers

def row10151_large_right_left : List Nat := row10151_block_009 ++ row10151_block_010
theorem row10151_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10151 row10151_large_right_left :=
  Budget.RowUpperBound.branch row10151_block_009_covers row10151_block_010_covers

def row10151_large_right_right_right : List Nat := row10151_block_012 ++ row10151_block_013
theorem row10151_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10151 row10151_large_right_right_right :=
  Budget.RowUpperBound.branch row10151_block_012_covers row10151_block_013_covers

def row10151_large_right_right : List Nat := row10151_block_011 ++ row10151_large_right_right_right
theorem row10151_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10151 row10151_large_right_right :=
  Budget.RowUpperBound.branch row10151_block_011_covers row10151_large_right_right_right_covers

def row10151_large_right : List Nat := row10151_large_right_left ++ row10151_large_right_right
theorem row10151_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10151 row10151_large_right :=
  Budget.RowUpperBound.branch row10151_large_right_left_covers row10151_large_right_right_covers

def row10151_large : List Nat := row10151_large_left ++ row10151_large_right
theorem row10151_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10151 row10151_large :=
  Budget.RowUpperBound.branch row10151_large_left_covers row10151_large_right_covers

def row10151 : List Nat := row10151_large
theorem row10151_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10151 row10151 := row10151_large_covers

end ProjectEulerSolutions.PrimePairLarge
