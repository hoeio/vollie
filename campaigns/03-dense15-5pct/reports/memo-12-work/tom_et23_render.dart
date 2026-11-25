// E-T23 (Tom): render prompts with the app's own chat-template engine (llamadart, as vendored in lab/app-c3), one user
// message each, assistant turn opened, metadata = the GGUF's chat template only (the files carry no
// tokenizer.ggml.bos_token / eos_token keys, so the engine uses its defaults, as the app does). *host*, no model.
// Input and output lines use c3_head_dump's escapes ("\n", "\\"). Run from lab/app-c3:
//   dart run --packages=.dart_tool/package_config.json <this file> <template.txt> <prompts.txt> <out.txt>
import 'dart:io';
import 'package:llamadart/src/core/template/chat_template_engine.dart';
import 'package:llamadart/llamadart.dart' show LlamaChatMessage, LlamaChatRole;

String unesc(String s) {
  final o = StringBuffer();
  for (var i = 0; i < s.length; i++) {
    if (s[i] == '\\' && i + 1 < s.length) { final c = s[++i]; o.write(c == 'n' ? '\n' : c == 't' ? '\t' : c); } else { o.write(s[i]); }
  }
  return o.toString();
}

void main(List<String> a) {
  final md = {'tokenizer.chat_template': File(a[0]).readAsStringSync()};
  final out = <String>[];
  for (final l in File(a[1]).readAsLinesSync()) {
    final r = ChatTemplateEngine.render(templateSource: md['tokenizer.chat_template'],
        messages: [LlamaChatMessage.fromText(role: LlamaChatRole.user, text: unesc(l))], metadata: md, addAssistant: true, enableThinking: false);
    out.add(r.prompt.replaceAll('\\', '\\\\').replaceAll('\n', '\\n'));
  }
  File(a[2]).writeAsStringSync(out.map((x) => '$x\n').join());
  stderr.writeln('rendered ${out.length} prompts; first begins ${out.first.substring(0, 40)}');
}
