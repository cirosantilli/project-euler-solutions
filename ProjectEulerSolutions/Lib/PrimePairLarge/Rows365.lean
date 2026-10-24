import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12241_small : List Nat := []
theorem row12241_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12241 row12241_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12241_block_005 : List Nat := []
theorem row12241_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12241 row12241_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12241_block_006 : List Nat := []
def plan12241_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row12241_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12241 row12241_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12241_block_006
  · decide
  · rfl

def row12241_block_007 : List Nat := [13003, 13591]
def plan12241_block_007 : List Budget.Choice := [(.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 17), (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 37), (.secondFactor 23), .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 19), (.secondFactor 7), (.secondFactor 37), .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12241_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12241 row12241_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12241_block_007
  · decide
  · rfl

def row12241_block_008 : List Nat := []
theorem row12241_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12241 row12241_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12241_block_009 : List Nat := []
theorem row12241_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12241 row12241_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12241_block_010 : List Nat := []
theorem row12241_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12241 row12241_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12241_block_011 : List Nat := []
theorem row12241_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12241 row12241_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12241_block_012 : List Nat := []
theorem row12241_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12241 row12241_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12241_block_013 : List Nat := []
theorem row12241_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12241 row12241_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12241_large_left_left : List Nat := row12241_small ++ row12241_block_005
theorem row12241_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12241 row12241_large_left_left :=
  Budget.RowUpperBound.branch row12241_small_covers row12241_block_005_covers

def row12241_large_left_right_right : List Nat := row12241_block_007 ++ row12241_block_008
theorem row12241_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12241 row12241_large_left_right_right :=
  Budget.RowUpperBound.branch row12241_block_007_covers row12241_block_008_covers

def row12241_large_left_right : List Nat := row12241_block_006 ++ row12241_large_left_right_right
theorem row12241_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12241 row12241_large_left_right :=
  Budget.RowUpperBound.branch row12241_block_006_covers row12241_large_left_right_right_covers

def row12241_large_left : List Nat := row12241_large_left_left ++ row12241_large_left_right
theorem row12241_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12241 row12241_large_left :=
  Budget.RowUpperBound.branch row12241_large_left_left_covers row12241_large_left_right_covers

def row12241_large_right_left : List Nat := row12241_block_009 ++ row12241_block_010
theorem row12241_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12241 row12241_large_right_left :=
  Budget.RowUpperBound.branch row12241_block_009_covers row12241_block_010_covers

def row12241_large_right_right_right : List Nat := row12241_block_012 ++ row12241_block_013
theorem row12241_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12241 row12241_large_right_right_right :=
  Budget.RowUpperBound.branch row12241_block_012_covers row12241_block_013_covers

def row12241_large_right_right : List Nat := row12241_block_011 ++ row12241_large_right_right_right
theorem row12241_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12241 row12241_large_right_right :=
  Budget.RowUpperBound.branch row12241_block_011_covers row12241_large_right_right_right_covers

def row12241_large_right : List Nat := row12241_large_right_left ++ row12241_large_right_right
theorem row12241_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12241 row12241_large_right :=
  Budget.RowUpperBound.branch row12241_large_right_left_covers row12241_large_right_right_covers

def row12241_large : List Nat := row12241_large_left ++ row12241_large_right
theorem row12241_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12241 row12241_large :=
  Budget.RowUpperBound.branch row12241_large_left_covers row12241_large_right_covers

def row12241 : List Nat := row12241_large
theorem row12241_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12241 row12241 := row12241_large_covers

def row12251_small : List Nat := []
theorem row12251_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12251 row12251_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12251_block_005 : List Nat := []
theorem row12251_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12251 row12251_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12251_block_006 : List Nat := []
def plan12251_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .second, (.secondFactor 19), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row12251_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12251 row12251_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12251_block_006
  · decide
  · rfl

def row12251_block_007 : List Nat := [12893, 13037, 13151, 13187]
def plan12251_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .second, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 23), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .second, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, .possible, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12251_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12251 row12251_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12251_block_007
  · decide
  · rfl

def row12251_block_008 : List Nat := []
theorem row12251_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12251 row12251_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12251_block_009 : List Nat := []
theorem row12251_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12251 row12251_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12251_block_010 : List Nat := []
theorem row12251_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12251 row12251_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12251_block_011 : List Nat := []
theorem row12251_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12251 row12251_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12251_block_012 : List Nat := []
theorem row12251_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12251 row12251_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12251_block_013 : List Nat := []
theorem row12251_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12251 row12251_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12251_large_left_left : List Nat := row12251_small ++ row12251_block_005
theorem row12251_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12251 row12251_large_left_left :=
  Budget.RowUpperBound.branch row12251_small_covers row12251_block_005_covers

def row12251_large_left_right_right : List Nat := row12251_block_007 ++ row12251_block_008
theorem row12251_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12251 row12251_large_left_right_right :=
  Budget.RowUpperBound.branch row12251_block_007_covers row12251_block_008_covers

def row12251_large_left_right : List Nat := row12251_block_006 ++ row12251_large_left_right_right
theorem row12251_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12251 row12251_large_left_right :=
  Budget.RowUpperBound.branch row12251_block_006_covers row12251_large_left_right_right_covers

def row12251_large_left : List Nat := row12251_large_left_left ++ row12251_large_left_right
theorem row12251_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12251 row12251_large_left :=
  Budget.RowUpperBound.branch row12251_large_left_left_covers row12251_large_left_right_covers

def row12251_large_right_left : List Nat := row12251_block_009 ++ row12251_block_010
theorem row12251_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12251 row12251_large_right_left :=
  Budget.RowUpperBound.branch row12251_block_009_covers row12251_block_010_covers

def row12251_large_right_right_right : List Nat := row12251_block_012 ++ row12251_block_013
theorem row12251_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12251 row12251_large_right_right_right :=
  Budget.RowUpperBound.branch row12251_block_012_covers row12251_block_013_covers

def row12251_large_right_right : List Nat := row12251_block_011 ++ row12251_large_right_right_right
theorem row12251_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12251 row12251_large_right_right :=
  Budget.RowUpperBound.branch row12251_block_011_covers row12251_large_right_right_right_covers

def row12251_large_right : List Nat := row12251_large_right_left ++ row12251_large_right_right
theorem row12251_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12251 row12251_large_right :=
  Budget.RowUpperBound.branch row12251_large_right_left_covers row12251_large_right_right_covers

def row12251_large : List Nat := row12251_large_left ++ row12251_large_right
theorem row12251_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12251 row12251_large :=
  Budget.RowUpperBound.branch row12251_large_left_covers row12251_large_right_covers

def row12251 : List Nat := row12251_large
theorem row12251_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12251 row12251 := row12251_large_covers

def row12253_small : List Nat := []
theorem row12253_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12253 row12253_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12253_block_005 : List Nat := []
theorem row12253_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12253 row12253_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12253_block_006 : List Nat := []
def plan12253_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 29), (.secondFactor 31), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row12253_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12253 row12253_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12253_block_006
  · decide
  · rfl

def row12253_block_007 : List Nat := [12721, 13183]
def plan12253_block_007 : List Budget.Choice := [(.secondFactor 29), (.secondFactor 19), .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 31), (.secondFactor 19), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .second, (.secondFactor 37), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .second, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12253_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12253 row12253_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12253_block_007
  · decide
  · rfl

def row12253_block_008 : List Nat := []
theorem row12253_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12253 row12253_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12253_block_009 : List Nat := []
theorem row12253_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12253 row12253_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12253_block_010 : List Nat := []
theorem row12253_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12253 row12253_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12253_block_011 : List Nat := []
theorem row12253_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12253 row12253_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12253_block_012 : List Nat := []
theorem row12253_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12253 row12253_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12253_block_013 : List Nat := []
theorem row12253_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12253 row12253_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12253_large_left_left : List Nat := row12253_small ++ row12253_block_005
theorem row12253_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12253 row12253_large_left_left :=
  Budget.RowUpperBound.branch row12253_small_covers row12253_block_005_covers

def row12253_large_left_right_right : List Nat := row12253_block_007 ++ row12253_block_008
theorem row12253_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12253 row12253_large_left_right_right :=
  Budget.RowUpperBound.branch row12253_block_007_covers row12253_block_008_covers

def row12253_large_left_right : List Nat := row12253_block_006 ++ row12253_large_left_right_right
theorem row12253_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12253 row12253_large_left_right :=
  Budget.RowUpperBound.branch row12253_block_006_covers row12253_large_left_right_right_covers

def row12253_large_left : List Nat := row12253_large_left_left ++ row12253_large_left_right
theorem row12253_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12253 row12253_large_left :=
  Budget.RowUpperBound.branch row12253_large_left_left_covers row12253_large_left_right_covers

def row12253_large_right_left : List Nat := row12253_block_009 ++ row12253_block_010
theorem row12253_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12253 row12253_large_right_left :=
  Budget.RowUpperBound.branch row12253_block_009_covers row12253_block_010_covers

def row12253_large_right_right_right : List Nat := row12253_block_012 ++ row12253_block_013
theorem row12253_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12253 row12253_large_right_right_right :=
  Budget.RowUpperBound.branch row12253_block_012_covers row12253_block_013_covers

def row12253_large_right_right : List Nat := row12253_block_011 ++ row12253_large_right_right_right
theorem row12253_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12253 row12253_large_right_right :=
  Budget.RowUpperBound.branch row12253_block_011_covers row12253_large_right_right_right_covers

def row12253_large_right : List Nat := row12253_large_right_left ++ row12253_large_right_right
theorem row12253_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12253 row12253_large_right :=
  Budget.RowUpperBound.branch row12253_large_right_left_covers row12253_large_right_right_covers

def row12253_large : List Nat := row12253_large_left ++ row12253_large_right
theorem row12253_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12253 row12253_large :=
  Budget.RowUpperBound.branch row12253_large_left_covers row12253_large_right_covers

def row12253 : List Nat := row12253_large
theorem row12253_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12253 row12253 := row12253_large_covers

def row12263_small : List Nat := []
theorem row12263_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12263 row12263_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12263_block_005 : List Nat := []
theorem row12263_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12263 row12263_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12263_block_006 : List Nat := []
def plan12263_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row12263_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12263 row12263_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12263_block_006
  · decide
  · rfl

def row12263_block_007 : List Nat := [12917, 13721, 13763]
def plan12263_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .second, (.firstFactor 7), .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .possible, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 17), (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 13), .first, .first, .possible, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12263_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12263 row12263_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12263_block_007
  · decide
  · rfl

def row12263_block_008 : List Nat := []
theorem row12263_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12263 row12263_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12263_block_009 : List Nat := []
theorem row12263_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12263 row12263_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12263_block_010 : List Nat := []
theorem row12263_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12263 row12263_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12263_block_011 : List Nat := []
theorem row12263_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12263 row12263_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12263_block_012 : List Nat := []
theorem row12263_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12263 row12263_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12263_block_013 : List Nat := []
theorem row12263_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12263 row12263_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12263_large_left_left : List Nat := row12263_small ++ row12263_block_005
theorem row12263_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12263 row12263_large_left_left :=
  Budget.RowUpperBound.branch row12263_small_covers row12263_block_005_covers

def row12263_large_left_right_right : List Nat := row12263_block_007 ++ row12263_block_008
theorem row12263_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12263 row12263_large_left_right_right :=
  Budget.RowUpperBound.branch row12263_block_007_covers row12263_block_008_covers

def row12263_large_left_right : List Nat := row12263_block_006 ++ row12263_large_left_right_right
theorem row12263_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12263 row12263_large_left_right :=
  Budget.RowUpperBound.branch row12263_block_006_covers row12263_large_left_right_right_covers

def row12263_large_left : List Nat := row12263_large_left_left ++ row12263_large_left_right
theorem row12263_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12263 row12263_large_left :=
  Budget.RowUpperBound.branch row12263_large_left_left_covers row12263_large_left_right_covers

def row12263_large_right_left : List Nat := row12263_block_009 ++ row12263_block_010
theorem row12263_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12263 row12263_large_right_left :=
  Budget.RowUpperBound.branch row12263_block_009_covers row12263_block_010_covers

def row12263_large_right_right_right : List Nat := row12263_block_012 ++ row12263_block_013
theorem row12263_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12263 row12263_large_right_right_right :=
  Budget.RowUpperBound.branch row12263_block_012_covers row12263_block_013_covers

def row12263_large_right_right : List Nat := row12263_block_011 ++ row12263_large_right_right_right
theorem row12263_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12263 row12263_large_right_right :=
  Budget.RowUpperBound.branch row12263_block_011_covers row12263_large_right_right_right_covers

def row12263_large_right : List Nat := row12263_large_right_left ++ row12263_large_right_right
theorem row12263_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12263 row12263_large_right :=
  Budget.RowUpperBound.branch row12263_large_right_left_covers row12263_large_right_right_covers

def row12263_large : List Nat := row12263_large_left ++ row12263_large_right
theorem row12263_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12263 row12263_large :=
  Budget.RowUpperBound.branch row12263_large_left_covers row12263_large_right_covers

def row12263 : List Nat := row12263_large
theorem row12263_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12263 row12263 := row12263_large_covers

end ProjectEulerSolutions.PrimePairLarge
