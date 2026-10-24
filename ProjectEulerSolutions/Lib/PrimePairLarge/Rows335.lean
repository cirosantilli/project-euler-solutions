import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11083_small : List Nat := []
theorem row11083_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11083 row11083_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11083_block_005 : List Nat := []
theorem row11083_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11083 row11083_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11083_block_006 : List Nat := [11353, 11587, 12211]
def plan11083_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 29), (.firstFactor 17), .first, .first, .second, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 13), .first, (.secondFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11083_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11083 row11083_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11083_block_006
  · decide
  · rfl

def row11083_block_007 : List Nat := []
def plan11083_block_007 : List Budget.Choice := [.first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, (.secondFactor 13), .first, .second, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 17), (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 37), (.secondFactor 31), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row11083_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11083 row11083_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11083_block_007
  · decide
  · rfl

def row11083_block_008 : List Nat := [14923]
def plan11083_block_008 : List Budget.Choice := [(.firstFactor 19), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .possible, (.secondFactor 17), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11083_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11083 row11083_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11083_block_008
  · decide
  · rfl

def row11083_block_009 : List Nat := []
theorem row11083_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11083 row11083_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11083_block_010 : List Nat := []
theorem row11083_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11083 row11083_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11083_block_011 : List Nat := []
theorem row11083_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11083 row11083_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11083_block_012 : List Nat := []
theorem row11083_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11083 row11083_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11083_block_013 : List Nat := []
theorem row11083_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11083 row11083_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11083_large_left_left : List Nat := row11083_small ++ row11083_block_005
theorem row11083_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11083 row11083_large_left_left :=
  Budget.RowUpperBound.branch row11083_small_covers row11083_block_005_covers

def row11083_large_left_right_right : List Nat := row11083_block_007 ++ row11083_block_008
theorem row11083_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11083 row11083_large_left_right_right :=
  Budget.RowUpperBound.branch row11083_block_007_covers row11083_block_008_covers

def row11083_large_left_right : List Nat := row11083_block_006 ++ row11083_large_left_right_right
theorem row11083_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11083 row11083_large_left_right :=
  Budget.RowUpperBound.branch row11083_block_006_covers row11083_large_left_right_right_covers

def row11083_large_left : List Nat := row11083_large_left_left ++ row11083_large_left_right
theorem row11083_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11083 row11083_large_left :=
  Budget.RowUpperBound.branch row11083_large_left_left_covers row11083_large_left_right_covers

def row11083_large_right_left : List Nat := row11083_block_009 ++ row11083_block_010
theorem row11083_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11083 row11083_large_right_left :=
  Budget.RowUpperBound.branch row11083_block_009_covers row11083_block_010_covers

def row11083_large_right_right_right : List Nat := row11083_block_012 ++ row11083_block_013
theorem row11083_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11083 row11083_large_right_right_right :=
  Budget.RowUpperBound.branch row11083_block_012_covers row11083_block_013_covers

def row11083_large_right_right : List Nat := row11083_block_011 ++ row11083_large_right_right_right
theorem row11083_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11083 row11083_large_right_right :=
  Budget.RowUpperBound.branch row11083_block_011_covers row11083_large_right_right_right_covers

def row11083_large_right : List Nat := row11083_large_right_left ++ row11083_large_right_right
theorem row11083_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11083 row11083_large_right :=
  Budget.RowUpperBound.branch row11083_large_right_left_covers row11083_large_right_right_covers

def row11083_large : List Nat := row11083_large_left ++ row11083_large_right
theorem row11083_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11083 row11083_large :=
  Budget.RowUpperBound.branch row11083_large_left_covers row11083_large_right_covers

def row11083 : List Nat := row11083_large
theorem row11083_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11083 row11083 := row11083_large_covers

def row11087_small : List Nat := []
theorem row11087_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11087 row11087_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11087_block_005 : List Nat := []
theorem row11087_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11087 row11087_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11087_block_006 : List Nat := [11867, 11939]
def plan11087_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .second, (.secondFactor 7), .second, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 37), .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .possible, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row11087_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11087 row11087_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11087_block_006
  · decide
  · rfl

def row11087_block_007 : List Nat := [12959, 13049]
def plan11087_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 37), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 17), .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.secondFactor 37), .possible, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 37), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11087_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11087 row11087_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11087_block_007
  · decide
  · rfl

def row11087_block_008 : List Nat := [14561]
def plan11087_block_008 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 19), .second, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11087_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11087 row11087_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11087_block_008
  · decide
  · rfl

def row11087_block_009 : List Nat := []
theorem row11087_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11087 row11087_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11087_block_010 : List Nat := []
theorem row11087_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11087 row11087_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11087_block_011 : List Nat := []
theorem row11087_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11087 row11087_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11087_block_012 : List Nat := []
theorem row11087_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11087 row11087_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11087_block_013 : List Nat := []
theorem row11087_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11087 row11087_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11087_large_left_left : List Nat := row11087_small ++ row11087_block_005
theorem row11087_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11087 row11087_large_left_left :=
  Budget.RowUpperBound.branch row11087_small_covers row11087_block_005_covers

def row11087_large_left_right_right : List Nat := row11087_block_007 ++ row11087_block_008
theorem row11087_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11087 row11087_large_left_right_right :=
  Budget.RowUpperBound.branch row11087_block_007_covers row11087_block_008_covers

def row11087_large_left_right : List Nat := row11087_block_006 ++ row11087_large_left_right_right
theorem row11087_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11087 row11087_large_left_right :=
  Budget.RowUpperBound.branch row11087_block_006_covers row11087_large_left_right_right_covers

def row11087_large_left : List Nat := row11087_large_left_left ++ row11087_large_left_right
theorem row11087_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11087 row11087_large_left :=
  Budget.RowUpperBound.branch row11087_large_left_left_covers row11087_large_left_right_covers

def row11087_large_right_left : List Nat := row11087_block_009 ++ row11087_block_010
theorem row11087_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11087 row11087_large_right_left :=
  Budget.RowUpperBound.branch row11087_block_009_covers row11087_block_010_covers

def row11087_large_right_right_right : List Nat := row11087_block_012 ++ row11087_block_013
theorem row11087_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11087 row11087_large_right_right_right :=
  Budget.RowUpperBound.branch row11087_block_012_covers row11087_block_013_covers

def row11087_large_right_right : List Nat := row11087_block_011 ++ row11087_large_right_right_right
theorem row11087_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11087 row11087_large_right_right :=
  Budget.RowUpperBound.branch row11087_block_011_covers row11087_large_right_right_right_covers

def row11087_large_right : List Nat := row11087_large_right_left ++ row11087_large_right_right
theorem row11087_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11087 row11087_large_right :=
  Budget.RowUpperBound.branch row11087_large_right_left_covers row11087_large_right_right_covers

def row11087_large : List Nat := row11087_large_left ++ row11087_large_right
theorem row11087_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11087 row11087_large :=
  Budget.RowUpperBound.branch row11087_large_left_covers row11087_large_right_covers

def row11087 : List Nat := row11087_large
theorem row11087_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11087 row11087 := row11087_large_covers

def row11093_small : List Nat := []
theorem row11093_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11093 row11093_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11093_block_005 : List Nat := []
theorem row11093_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11093 row11093_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11093_block_006 : List Nat := [11981, 12143]
def plan11093_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 17), (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 31), .first, (.secondFactor 31), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row11093_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11093 row11093_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11093_block_006
  · decide
  · rfl

def row11093_block_007 : List Nat := [13151, 13367, 14303]
def plan11093_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 29), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 37), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 17), .first, .second, (.firstFactor 11), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 31), (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 23), (.firstFactor 7), .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, (.firstFactor 37), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .second, .first, (.firstFactor 17), .first, .first, .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row11093_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11093 row11093_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11093_block_007
  · decide
  · rfl

def row11093_block_008 : List Nat := [14813]
def plan11093_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .possible, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11093_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11093 row11093_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11093_block_008
  · decide
  · rfl

def row11093_block_009 : List Nat := []
theorem row11093_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11093 row11093_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11093_block_010 : List Nat := []
theorem row11093_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11093 row11093_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11093_block_011 : List Nat := []
theorem row11093_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11093 row11093_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11093_block_012 : List Nat := []
theorem row11093_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11093 row11093_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11093_block_013 : List Nat := []
theorem row11093_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11093 row11093_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11093_large_left_left : List Nat := row11093_small ++ row11093_block_005
theorem row11093_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11093 row11093_large_left_left :=
  Budget.RowUpperBound.branch row11093_small_covers row11093_block_005_covers

def row11093_large_left_right_right : List Nat := row11093_block_007 ++ row11093_block_008
theorem row11093_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11093 row11093_large_left_right_right :=
  Budget.RowUpperBound.branch row11093_block_007_covers row11093_block_008_covers

def row11093_large_left_right : List Nat := row11093_block_006 ++ row11093_large_left_right_right
theorem row11093_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11093 row11093_large_left_right :=
  Budget.RowUpperBound.branch row11093_block_006_covers row11093_large_left_right_right_covers

def row11093_large_left : List Nat := row11093_large_left_left ++ row11093_large_left_right
theorem row11093_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11093 row11093_large_left :=
  Budget.RowUpperBound.branch row11093_large_left_left_covers row11093_large_left_right_covers

def row11093_large_right_left : List Nat := row11093_block_009 ++ row11093_block_010
theorem row11093_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11093 row11093_large_right_left :=
  Budget.RowUpperBound.branch row11093_block_009_covers row11093_block_010_covers

def row11093_large_right_right_right : List Nat := row11093_block_012 ++ row11093_block_013
theorem row11093_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11093 row11093_large_right_right_right :=
  Budget.RowUpperBound.branch row11093_block_012_covers row11093_block_013_covers

def row11093_large_right_right : List Nat := row11093_block_011 ++ row11093_large_right_right_right
theorem row11093_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11093 row11093_large_right_right :=
  Budget.RowUpperBound.branch row11093_block_011_covers row11093_large_right_right_right_covers

def row11093_large_right : List Nat := row11093_large_right_left ++ row11093_large_right_right
theorem row11093_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11093 row11093_large_right :=
  Budget.RowUpperBound.branch row11093_large_right_left_covers row11093_large_right_right_covers

def row11093_large : List Nat := row11093_large_left ++ row11093_large_right
theorem row11093_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11093 row11093_large :=
  Budget.RowUpperBound.branch row11093_large_left_covers row11093_large_right_covers

def row11093 : List Nat := row11093_large
theorem row11093_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11093 row11093 := row11093_large_covers

def row11113_small : List Nat := []
theorem row11113_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11113 row11113_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11113_block_005 : List Nat := []
theorem row11113_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11113 row11113_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11113_block_006 : List Nat := []
def plan11113_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 37), .first, .first, .second, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 37), (.firstFactor 17), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row11113_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11113 row11113_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11113_block_006
  · decide
  · rfl

def row11113_block_007 : List Nat := [12739, 12841, 13873, 14419]
def plan11113_block_007 : List Budget.Choice := [(.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .second, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .second, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 11), .second, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 37), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 29), .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .second, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11113_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11113 row11113_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11113_block_007
  · decide
  · rfl

def row11113_block_008 : List Nat := []
def plan11113_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11113_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11113 row11113_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11113_block_008
  · decide
  · rfl

def row11113_block_009 : List Nat := []
theorem row11113_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11113 row11113_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11113_block_010 : List Nat := []
theorem row11113_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11113 row11113_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11113_block_011 : List Nat := []
theorem row11113_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11113 row11113_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11113_block_012 : List Nat := []
theorem row11113_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11113 row11113_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11113_block_013 : List Nat := []
theorem row11113_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11113 row11113_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11113_large_left_left : List Nat := row11113_small ++ row11113_block_005
theorem row11113_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11113 row11113_large_left_left :=
  Budget.RowUpperBound.branch row11113_small_covers row11113_block_005_covers

def row11113_large_left_right_right : List Nat := row11113_block_007 ++ row11113_block_008
theorem row11113_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11113 row11113_large_left_right_right :=
  Budget.RowUpperBound.branch row11113_block_007_covers row11113_block_008_covers

def row11113_large_left_right : List Nat := row11113_block_006 ++ row11113_large_left_right_right
theorem row11113_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11113 row11113_large_left_right :=
  Budget.RowUpperBound.branch row11113_block_006_covers row11113_large_left_right_right_covers

def row11113_large_left : List Nat := row11113_large_left_left ++ row11113_large_left_right
theorem row11113_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11113 row11113_large_left :=
  Budget.RowUpperBound.branch row11113_large_left_left_covers row11113_large_left_right_covers

def row11113_large_right_left : List Nat := row11113_block_009 ++ row11113_block_010
theorem row11113_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11113 row11113_large_right_left :=
  Budget.RowUpperBound.branch row11113_block_009_covers row11113_block_010_covers

def row11113_large_right_right_right : List Nat := row11113_block_012 ++ row11113_block_013
theorem row11113_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11113 row11113_large_right_right_right :=
  Budget.RowUpperBound.branch row11113_block_012_covers row11113_block_013_covers

def row11113_large_right_right : List Nat := row11113_block_011 ++ row11113_large_right_right_right
theorem row11113_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11113 row11113_large_right_right :=
  Budget.RowUpperBound.branch row11113_block_011_covers row11113_large_right_right_right_covers

def row11113_large_right : List Nat := row11113_large_right_left ++ row11113_large_right_right
theorem row11113_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11113 row11113_large_right :=
  Budget.RowUpperBound.branch row11113_large_right_left_covers row11113_large_right_right_covers

def row11113_large : List Nat := row11113_large_left ++ row11113_large_right
theorem row11113_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11113 row11113_large :=
  Budget.RowUpperBound.branch row11113_large_left_covers row11113_large_right_covers

def row11113 : List Nat := row11113_large
theorem row11113_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11113 row11113 := row11113_large_covers

end ProjectEulerSolutions.PrimePairLarge
