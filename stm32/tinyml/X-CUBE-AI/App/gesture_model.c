/**
  ******************************************************************************
  * @file    gesture_model.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-09-23T20:01:37+0530
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "gesture_model.h"
#include "gesture_model_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_gesture_model
 
#undef AI_GESTURE_MODEL_MODEL_SIGNATURE
#define AI_GESTURE_MODEL_MODEL_SIGNATURE     "0x297c579af01d228b9025e5739933fcd0"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-09-23T20:01:37+0530"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_GESTURE_MODEL_N_BATCHES
#define AI_GESTURE_MODEL_N_BATCHES         (1)

static ai_ptr g_gesture_model_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_gesture_model_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input_layer0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 300, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3072, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  pool_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2816, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  pool_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1408, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  nl_18_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24576, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 180224, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3272, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7680, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  gemm_16_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 2048, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  gemm_17_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 128, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003273638430982828f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0013534208992496133f, 0.0014201912563294172f, 0.0014722869964316487f, 0.0014585522003471851f, 0.001515398034825921f, 0.0013437338639050722f, 0.0012865173630416393f, 0.0011474167695268989f, 0.0014506250154227018f, 0.0011151951039209962f, 0.0013944546226412058f, 0.0012164427898824215f, 0.0013141384115442634f, 0.001449001021683216f, 0.0014410599833354354f, 0.0014339843764901161f, 0.001462167943827808f, 0.0014333140570670366f, 0.00121738959569484f, 0.001313262851908803f, 0.0012007267214357853f, 0.0012503276811912656f, 0.0014056680956855416f, 0.0013544111279770732f, 0.0012712483294308186f, 0.001147400471381843f, 0.0012527204817160964f, 0.0011833481257781386f, 0.0014484694693237543f, 0.0012918425491079688f, 0.0013305871980264783f, 0.0014655626146122813f, 0.0014555164379999042f, 0.0013992515159770846f, 0.0013638115487992764f, 0.0013685979647561908f, 0.0013066502287983894f, 0.0011796324979513884f, 0.0014697167789563537f, 0.0013691475614905357f, 0.0014300276525318623f, 0.0013553149765357375f, 0.0014489940367639065f, 0.0014330625999718904f, 0.0013786269119009376f, 0.001402034074999392f, 0.0014839237555861473f, 0.0012315382482483983f, 0.0013936886098235846f, 0.001332358457148075f, 0.001406469731591642f, 0.0014496288495138288f, 0.0013060083147138357f, 0.0013276843819767237f, 0.0012942487373948097f, 0.0012361211702227592f, 0.0012876719702035189f, 0.0015001080464571714f, 0.0014095822116360068f, 0.0013750180369243026f, 0.0013867862289771438f, 0.0013911983696743846f, 0.0014790988061577082f, 0.0013151505263522267f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0040960656479001045f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0009911081288009882f, 0.0009038879070430994f, 0.0009372651111334562f, 0.0008464209386147559f, 0.00088968884665519f, 0.0008869667071849108f, 0.0008901405963115394f, 0.0009763356647454202f, 0.0009487585630267859f, 0.000988046289421618f, 0.000981464283540845f, 0.0009826001478359103f, 0.0009094728156924248f, 0.0008594035170972347f, 0.0008827002020552754f, 0.000997978146187961f, 0.0009593137656338513f, 0.0009778323583304882f, 0.0009234017925336957f, 0.0009335328941233456f, 0.0009148558019660413f, 0.0008502617129124701f, 0.00085588323418051f, 0.000862243352457881f, 0.0009708883590064943f, 0.0009408220648765564f, 0.0009196860482916236f, 0.0009644374949857593f, 0.000926500535570085f, 0.0009020210709422827f, 0.0009484363836236298f, 0.0008747130050323904f, 0.000939058605581522f, 0.0009426332544535398f, 0.0009061815799213946f, 0.000982536468654871f, 0.0009870410431176424f, 0.0009421283612027764f, 0.0009323896956630051f, 0.000937983684707433f, 0.0008915451471693814f, 0.0009817896643653512f, 0.0009562757913954556f, 0.0008682020124979317f, 0.000898710684850812f, 0.0009762412519194186f, 0.000940614496357739f, 0.0009240235085599124f, 0.0008538819383829832f, 0.0008983677835203707f, 0.0009502117754891515f, 0.0009190791752189398f, 0.0009170828852802515f, 0.0008821784285828471f, 0.000996435759589076f, 0.000969509594142437f, 0.000967418949585408f, 0.0008330394048243761f, 0.0009759454987943172f, 0.0009489505318924785f, 0.0010082387598231435f, 0.0009241941152140498f, 0.0009196570026688278f, 0.0008814655593596399f, 0.0008615676779299974f, 0.0009074521367438138f, 0.0009945851052179933f, 0.0009711678139865398f, 0.0009514091652818024f, 0.000915133161470294f, 0.00096251314971596f, 0.0009039725409820676f, 0.0009467173949815333f, 0.0009114021668210626f, 0.0009534245473332703f, 0.000970206456258893f, 0.0009755789069458842f, 0.0008201568853110075f, 0.0009120448376052082f, 0.0008796267793513834f, 0.0008753183647058904f, 0.0009327073348686099f, 0.0008461653487756848f, 0.000921979604754597f, 0.0009659873321652412f, 0.0009463024325668812f, 0.0009325899300165474f, 0.000950440822634846f, 0.0009250238654203713f, 0.0009060694137588143f, 0.00094060436822474f, 0.0009527703514322639f, 0.0008411795715801418f, 0.000903177831787616f, 0.0008358046179637313f, 0.00095938058802858f, 0.0008689265232533216f, 0.0009377145324833691f, 0.0009484785259701312f, 0.0009792773053050041f, 0.0008374902536161244f, 0.0009417917462997139f, 0.0008401967934332788f, 0.0010094104800373316f, 0.0009412823710590601f, 0.000950075511354953f, 0.0009682902600616217f, 0.0008841915405355394f, 0.0009190217824652791f, 0.0008751415880396962f, 0.0009275428019464016f, 0.000963354657869786f, 0.000864653498865664f, 0.0009333146153949201f, 0.0009784064022824168f, 0.0009510698146186769f, 0.0009347101440653205f, 0.0008928156457841396f, 0.0009513446129858494f, 0.0009052509558387101f, 0.0009756061481311917f, 0.0008771599968895316f, 0.0009073673281818628f, 0.000945130770560354f, 0.0009205184760503471f, 0.0009025344043038785f, 0.0009524903725832701f, 0.0008424586849287152f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_16_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023535456508398056f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_16_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006587234674952924f, 0.0005378847708925605f, 0.0006640513893216848f, 0.0006586740491911769f, 0.0006731839966960251f, 0.0005573813687078655f, 0.0006618918268941343f, 0.0004916823236271739f, 0.0006542186019942164f, 0.0006529949023388326f, 0.0005652321269735694f, 0.0006538056186400354f, 0.0006349020404741168f, 0.0005371565348468721f, 0.0005447192233987153f, 0.0004920040955767035f, 0.000657274853438139f, 0.0006795168737880886f, 0.0006664277752861381f, 0.0005505149019882083f, 0.0006844527088105679f, 0.0006592639838345349f, 0.0006431273650377989f, 0.000674000708386302f, 0.0006657211924903095f, 0.0005708052194677293f, 0.0005695578292943537f, 0.0006411291542463005f, 0.0005388253484852612f, 0.000679361866787076f, 0.0006659195641987026f, 0.0006497495342046022f, 0.0006713615148328245f, 0.0006586348172277212f, 0.0005389860016293824f, 0.0006723927217535675f, 0.000537901883944869f, 0.0006579702603630722f, 0.0006719625671394169f, 0.0006519406451843679f, 0.0006418183911591768f, 0.0006462763412855566f, 0.0006577371968887746f, 0.0006397571414709091f, 0.000643754203338176f, 0.0006527426303364336f, 0.0005531838978640735f, 0.0006313786143437028f, 0.0005299169570207596f, 0.0006433342932723463f, 0.0006876704865135252f, 0.0006531492690555751f, 0.0005750301643274724f, 0.0006765457219444215f, 0.0006665482651442289f, 0.0005504140281118453f, 0.0005383125972002745f, 0.0006849052151665092f, 0.0005389420548453927f, 0.0005557521944865584f, 0.0006631512078456581f, 0.0006599150947295129f, 0.0005541688296943903f, 0.0005308522959239781f, 0.0005346619873307645f, 0.0006646988331340253f, 0.0006625722162425518f, 0.0005553604569286108f, 0.0006479181465692818f, 0.0006374529330059886f, 0.0006411876529455185f, 0.0006549023673869669f, 0.0006443662568926811f, 0.0006626301910728216f, 0.0005345707759261131f, 0.0006771715125069022f, 0.0006560073234140873f, 0.0006544689531438053f, 0.0006580430781468749f, 0.0005467737792059779f, 0.0005475831567309797f, 0.0006664724205620587f, 0.0006678958889096975f, 0.0006813702057115734f, 0.0005523333675228059f, 0.000658915494568646f, 0.0006654244498349726f, 0.0007011126726865768f, 0.0006483625620603561f, 0.0006639478378929198f, 0.0005599359865300357f, 0.0006611234857700765f, 0.0006587249226868153f, 0.00063309510005638f, 0.0006564510986208916f, 0.0006648307899013162f, 0.000654917093925178f, 0.000666084757540375f, 0.000577470229472965f, 0.0005586815532296896f, 0.0006614481681026518f, 0.0006540770991705358f, 0.0006598869222216308f, 0.0006151393754407763f, 0.0006328186136670411f, 0.000560514279641211f, 0.0005373568274080753f, 0.0005555161042138934f, 0.0006549764657393098f, 0.0006422232836484909f, 0.0006535034044645727f, 0.0005380006041377783f, 0.0006239725626073778f, 0.0006509546656161547f, 0.0006504287011921406f, 0.0005344076780602336f, 0.0006556756561622024f, 0.000554811442270875f, 0.0005505515728145838f, 0.0006563428323715925f, 0.00065142463427037f, 0.0006572656566277146f, 0.0006543988129124045f, 0.0006580738117918372f, 0.0006857083644717932f, 0.0006444764439947903f, 0.0006462543387897313f, 0.0006351420888677239f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_17_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.20366497337818146f),
    AI_PACK_INTQ_ZP(30)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_17_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0018813915085047483f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_18_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0040960656479001045f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003273638430982828f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_input_layer0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-1)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 48, 1), AI_STRIDE_INIT(4, 1, 1, 64, 3072),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 3272, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3272, 3272),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 6, 3, 1, 64), AI_STRIDE_INIT(4, 1, 6, 384, 1152),
  1, &conv2d_1_weights_array, &conv2d_1_weights_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_7_bias_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 22, 1), AI_STRIDE_INIT(4, 1, 1, 128, 2816),
  1, &conv2d_7_output_array, &conv2d_7_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 7680, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7680, 7680),
  1, &conv2d_7_scratch0_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 1, 128), AI_STRIDE_INIT(4, 1, 64, 8192, 24576),
  1, &conv2d_7_weights_array, &conv2d_7_weights_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &gemm_16_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &gemm_16_output_array, &gemm_16_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_scratch0, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 2048, 1, 1), AI_STRIDE_INIT(4, 2, 2, 4096, 4096),
  1, &gemm_16_scratch0_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  gemm_16_weights, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1408, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1408, 180224, 180224),
  1, &gemm_16_weights_array, &gemm_16_weights_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_17_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_17_output_array, &gemm_17_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_scratch0, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 2, 2, 256, 256),
  1, &gemm_17_scratch0_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  gemm_17_weights, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 128, 1, 1, 1), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &gemm_17_weights_array, &gemm_17_weights_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  nl_18_output, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_18_output_array, &nl_18_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  pool_10_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 11, 1), AI_STRIDE_INIT(4, 1, 1, 128, 1408),
  1, &pool_10_output_array, &pool_10_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  pool_10_output0, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 1408, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1408, 1408),
  1, &pool_10_output_array, &pool_10_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  pool_4_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 24, 1), AI_STRIDE_INIT(4, 1, 1, 64, 1536),
  1, &pool_4_output_array, &pool_4_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_layer0_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 1, 50), AI_STRIDE_INIT(4, 1, 1, 6, 6),
  1, &serving_default_input_layer0_output_array, &serving_default_input_layer0_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_layer0_output0, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 50, 1), AI_STRIDE_INIT(4, 1, 1, 6, 300),
  1, &serving_default_input_layer0_output_array, &serving_default_input_layer0_output_array_intq)



/**  Layer declarations section  **********************************************/



AI_STATIC_CONST ai_i8 nl_18_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -126, -126, -126, -125, -124, -124, -123, -122, -120, -119, -116, -114, -111, -108, -103, -98, -93, -86, -78, -70, -60, -49, -38, -26, -13, 0, 13, 26, 38, 49, 60, 70, 78, 86, 93, 98, 103, 108, 111, 114, 116, 119, 120, 122, 123, 124, 124, 125, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_18_nl_params, AI_ARRAY_FORMAT_S8,
    nl_18_nl_params_data, nl_18_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_18_layer, 18,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_18_chain,
  NULL, &nl_18_layer, AI_STATIC, 
  .nl_params = &nl_18_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_17_weights, &gemm_17_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_17_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_17_layer, 17,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &gemm_17_chain,
  NULL, &nl_18_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_10_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_16_weights, &gemm_16_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_16_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_16_layer, 16,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_16_chain,
  NULL, &gemm_17_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_10_layer, 10,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_10_chain,
  NULL, &gemm_16_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &conv2d_7_chain,
  NULL, &pool_10_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_4_layer, 4,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &pool_4_chain,
  NULL, &conv2d_7_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input_layer0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
  &conv2d_1_chain,
  NULL, &pool_4_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 207364, 1, 1),
    207364, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 12032, 1, 1),
    12032, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_GESTURE_MODEL_IN_NUM, &serving_default_input_layer0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_GESTURE_MODEL_OUT_NUM, &nl_18_output),
  &conv2d_1_layer, 0x00d50d29, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 207364, 1, 1),
      207364, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 12032, 1, 1),
      12032, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_GESTURE_MODEL_IN_NUM, &serving_default_input_layer0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_GESTURE_MODEL_OUT_NUM, &nl_18_output),
  &conv2d_1_layer, 0x00d50d29, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool gesture_model_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_gesture_model_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_input_layer0_output_array.data = AI_PTR(g_gesture_model_activations_map[0] + 0);
    serving_default_input_layer0_output_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 0);
    conv2d_1_scratch0_array.data = AI_PTR(g_gesture_model_activations_map[0] + 300);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 300);
    conv2d_1_output_array.data = AI_PTR(g_gesture_model_activations_map[0] + 3572);
    conv2d_1_output_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 3572);
    pool_4_output_array.data = AI_PTR(g_gesture_model_activations_map[0] + 0);
    pool_4_output_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 0);
    conv2d_7_scratch0_array.data = AI_PTR(g_gesture_model_activations_map[0] + 1536);
    conv2d_7_scratch0_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 1536);
    conv2d_7_output_array.data = AI_PTR(g_gesture_model_activations_map[0] + 9216);
    conv2d_7_output_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 9216);
    pool_10_output_array.data = AI_PTR(g_gesture_model_activations_map[0] + 0);
    pool_10_output_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 0);
    gemm_16_scratch0_array.data = AI_PTR(g_gesture_model_activations_map[0] + 1408);
    gemm_16_scratch0_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 1408);
    gemm_16_output_array.data = AI_PTR(g_gesture_model_activations_map[0] + 5504);
    gemm_16_output_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 5504);
    gemm_17_scratch0_array.data = AI_PTR(g_gesture_model_activations_map[0] + 0);
    gemm_17_scratch0_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 0);
    gemm_17_output_array.data = AI_PTR(g_gesture_model_activations_map[0] + 256);
    gemm_17_output_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 256);
    nl_18_output_array.data = AI_PTR(g_gesture_model_activations_map[0] + 0);
    nl_18_output_array.data_start = AI_PTR(g_gesture_model_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool gesture_model_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_gesture_model_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(g_gesture_model_weights_map[0] + 0);
    conv2d_1_weights_array.data_start = AI_PTR(g_gesture_model_weights_map[0] + 0);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(g_gesture_model_weights_map[0] + 1152);
    conv2d_1_bias_array.data_start = AI_PTR(g_gesture_model_weights_map[0] + 1152);
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(g_gesture_model_weights_map[0] + 1408);
    conv2d_7_weights_array.data_start = AI_PTR(g_gesture_model_weights_map[0] + 1408);
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(g_gesture_model_weights_map[0] + 25984);
    conv2d_7_bias_array.data_start = AI_PTR(g_gesture_model_weights_map[0] + 25984);
    gemm_16_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_16_weights_array.data = AI_PTR(g_gesture_model_weights_map[0] + 26496);
    gemm_16_weights_array.data_start = AI_PTR(g_gesture_model_weights_map[0] + 26496);
    gemm_16_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_16_bias_array.data = AI_PTR(g_gesture_model_weights_map[0] + 206720);
    gemm_16_bias_array.data_start = AI_PTR(g_gesture_model_weights_map[0] + 206720);
    gemm_17_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_17_weights_array.data = AI_PTR(g_gesture_model_weights_map[0] + 207232);
    gemm_17_weights_array.data_start = AI_PTR(g_gesture_model_weights_map[0] + 207232);
    gemm_17_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_17_bias_array.data = AI_PTR(g_gesture_model_weights_map[0] + 207360);
    gemm_17_bias_array.data_start = AI_PTR(g_gesture_model_weights_map[0] + 207360);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_gesture_model_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_GESTURE_MODEL_MODEL_NAME,
      .model_signature   = AI_GESTURE_MODEL_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 782530,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x00d50d29,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_gesture_model_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_GESTURE_MODEL_MODEL_NAME,
      .model_signature   = AI_GESTURE_MODEL_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 782530,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x00d50d29,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_gesture_model_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_gesture_model_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_gesture_model_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_gesture_model_create(network, AI_GESTURE_MODEL_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_gesture_model_data_params_get(&params) != true) {
    err = ai_gesture_model_get_error(*network);
    return err;
  }
#if defined(AI_GESTURE_MODEL_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_GESTURE_MODEL_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_gesture_model_init(*network, &params) != true) {
    err = ai_gesture_model_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_gesture_model_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_gesture_model_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_gesture_model_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_gesture_model_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= gesture_model_configure_weights(net_ctx, params);
  ok &= gesture_model_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_gesture_model_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_gesture_model_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_GESTURE_MODEL_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

